from __future__ import annotations
import random
from typing import List, Optional, Tuple
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import lfilter
import argparse


class Kernel:
    def __init__(self, data: List[float]):
        self.data = data
        for i in range(len(self.data)):
            if np.isclose(self.data[i], 0.):
                self.data[i] = 0.
        self.compress()

    def _access_at(self, index: int):
        while len(self.data) <= index:
            self.data.append(0.)
    
    def __getitem__(self, index: int) -> float:
        self._access_at(index)
        return self.data[index]
    
    def __setitem__(self, index: int, value: float):
        self._access_at(index)
        value = 0. if np.isclose(value, 0.) else value
        self.data[index] = value

    def copy(self) -> 'Kernel':
        return Kernel(self.data.copy())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        self.compress()
        data = " ".join([f"{x:.8f}" for x in self.data])
        return f"Kernel([ {data} ])"

    def compress(self):
        """Remove trailing zeros from the signal."""
        while self.data[-1] == 0.0:
            self.data.pop()
    
    def num_terms(self) -> int:
        """Return the number of non-zero terms in the signal."""
        self.compress()
        return len(self.data) - self.data.count(0.)

class Filter:
    pass

class FIR:
    def __init__(self, b: Kernel):
        """
        Initialize FIR compiler with feedforward coefficients.
        
        Args:
            b: List of feedforward coefficients [b0, b1, b2, ...]
        
            Computes f(x) = b0*g(x) + b1*g(x-1) + ...
        """
        b.compress()
        self.b = b

    def __repr__(self) -> str:
        self.b.compress()
        return f"FIR(b={self.b})"
    
    def run(self, x: List[float]) -> List[float]:
        """Run the filter on the input signal x."""
        y = []
        for i in range(len(x)):
            res = 0.
            for j in range(len(self.b)):
                if i - j >= 0:
                    res += self.b[j] * x[i - j]
            y.append(res)
        return y

    def num_taps(self) -> int:
        """Return the number of taps in the filter."""
        self.b.compress()
        return len(self.b)

class IIR:
    def __init__(self, a: Kernel):
        """
        Initialize IIR compiler with feedback coefficients.
        
        Args:
            b: List of feedforward coefficients [b0, b1, b2, ...]
            a: List of feedback coefficients [a0, a1, a2, ...]
        
            Computes f(x) = a1*f(x-1) + a2*f(x-2) + ... + g(x)
            a0 must be 1.0
        """
        assert a[0] == 1.0, "First coefficient must be 1.0"
        a.compress()
        self.a = a

    def __repr__(self) -> str:
        self.a.compress()
        return f"IIR(a={self.a})"
    
    def reformulated(self, mask: List[bool] | str) -> Optional[Tuple[FIR, 'IIR']]:
        """
        Reformulate the IIR filter. The mask indicates which coefficients to keep,
        and the length of the mask indicates how farther back the reformulation can consider.
        For example, a mask with length 8 means we will consider coefficients up to 8 samples
        back (f(x), f(x-1), ..., f(x-7)).
        """
        if isinstance(mask, str):
            mask = [c == '1' for c in mask]

        self.a.compress()
        _mask = np.array(mask, dtype=bool)

        assert len(_mask) >= len(self.a), "Mask must be longer than a coefficients"

        len_a = len(_mask)
        len_b = len(_mask) + 1 # TODO: should this be one or zero?

        num_rewrites = len(_mask) - len(self.a) + 1
        A = np.zeros((len_a + len_b, num_rewrites))

        def make_rewrite(offset: int):
            rw_vec = [0.] * offset
            rw_vec.append(-1.)
            rw_vec.extend(self.a.data[1:])
            assert len(rw_vec) <= len_a, "Rewrite vector exceeds reserved length for a coefficients"
            rw_vec.extend([0.] * (len_a - len(rw_vec)))

            rw_vec.extend([0.] * offset)
            rw_vec.append(1.)
            assert len(rw_vec) <= len_a + len_b, "Rewrite vector exceeds reserved length for b coefficients"
            # TODO: Simplify the logic here
            rw_vec.extend([0.] * (len_a + len_b - len(rw_vec)))

            return np.array(rw_vec)

        for i in range(num_rewrites):
            A[:, i] = make_rewrite(i)

        assert not _mask[0], "The first coefficient must not be kept"

        extended_mask = np.concatenate((_mask, np.ones(len_b, dtype=bool)))
        A_zero_entries = A[~extended_mask, :]
        c = np.zeros(np.sum(~_mask))
        c[0] = -1.

        sol, residual = lstsq(A_zero_entries, c)[0:2]
        if sum(residual) > 1e-6:
            return None  # No solution found
        else:
            coeffs = A @ sol

            b = Kernel(coeffs[len_a:len_a + len_b].tolist())

            a = Kernel(coeffs[:len_a].tolist())
            assert np.isclose(a[0], -1)
            a[0] = 1.0  # Ensure the first coefficient is 1.0

            return (FIR(b), IIR(a))


    def delay(self, n: int) -> Tuple[FIR, 'IIR']:
        """Delay the IIR filter by n samples. This is not semantic-preserving."""
        # I'm doing a manual Gaussian elimination here, but it is possibel to do the same 
        # with reformulation.
        a = self.a.copy()
        b = Kernel([1.])
        for i in range(1, n):
            for j in range(1, len(a)):
                a[j + i] += a[i] * self.a[j]
            b[i] += a[i]
            a[i] = 0.
        return FIR(b), IIR(a)

    def dilate(self, lanes: int, step: int = 2) -> Tuple[List[FIR], 'IIR']:
        """Expand the filter into a tower of dilating filters."""
        assert lanes > 0 and (lanes & (lanes - 1)) == 0, "Lanes must be a power of 2"

        if step > lanes:
            return [], self
        order = 1
        dilating_filters = None
        while True:
            mask = [False] * (1 + order * step)
            for i in range(order):
                mask[(i + 1) * step] = True
            if len(mask) < len(self.a):
                mask.extend([False] * (len(self.a) - len(mask)))
            dilating_filters = self.reformulated(mask)
            if dilating_filters is None:
                order += 1
            else:
                fir, iir = dilating_filters
                rest_firs, iir = iir.dilate(lanes, step=step*2)
                return [fir] + rest_firs, iir

    def run(self, x: List[float]) -> List[float]:
        """Run the filter on the input signal x."""
        y: List[float] = []
        for i in range(len(x)):
            res = x[i]
            for j in range(1, len(self.a)):
                if i - j >= 0:
                    res += self.a[j] * y[i - j]
            y.append(res)
        return y
    
    def feedback_delay(self) -> int:
        for i in range(1, len(self.a)):
            if self.a[i] != 0.:
                return i
        raise ValueError("The filter is not recursive")

class IIRCompiler:
    filter: List[FIR | IIR]

    def __init__(self, filter: List[FIR | IIR]):
        self.filter = filter
    
    def _pos_to_idx(self, pos: int) -> int:
        return len(self.filter) + pos if pos < 0 else pos

    def delay(self, pos: int, n: int) -> None:
        """Delay the filter at position pos by n samples."""
        idx = self._pos_to_idx(pos)

        f = self.filter[idx]
        assert isinstance(f, IIR), "Can only delay IIR filters"
        fir, iir = f.delay(n)
        self.filter = self.filter[:idx] + [fir, iir] + self.filter[idx + 1:]

    def dilate(self, pos: int, lanes: int) -> None:
        """Expand the filter into a tower of dilating filters."""
        idx = self._pos_to_idx(pos)

        f = self.filter[idx]
        assert isinstance(f, IIR), "Can only dilate IIR filters"
        firs, iir = f.dilate(lanes)
        self.filter = self.filter[:idx] + firs + [iir] + self.filter[idx + 1:]

    def reformulate(self, pos: int, mask: List[bool] | str) -> None:
        """Reformulate the IIR filter at position pos."""
        idx = self._pos_to_idx(pos)

        f = self.filter[idx]
        assert isinstance(f, IIR), "Can only reformulate IIR filters"
        reformulated = f.reformulated(mask)
        assert reformulated is not None, "Reformulation failed"
        fir, iir = reformulated
        self.filter = self.filter[:idx] + [fir, iir] + self.filter[idx + 1:]

    def cost(self, max_lanes: int) -> float:
        cost = 0.
        for f in self.filter:
            if isinstance(f, FIR):
                cost += (f.b.num_terms() - 1) / max_lanes
            else:
                delay = f.feedback_delay()
                cost += (f.a.num_terms() - 1.) / min(max_lanes, delay)
        return cost

    def run(self, x: List[float]) -> List[float]:
        """Run the filter on the input signal x."""
        y = []
        for f in self.filter:
            y = f.run(x)
            x = y
        return y


def main():
    np.set_printoptions(precision=8)
    # filter = IIR(
    #     b=Kernel([1]),
    #     a=Kernel([1.0, 2, -1])
    # )
    filter = IIR(
        a=Kernel([1.0, 1.8, -0.9])
    )
    # filter = IIR(
    #     a=Kernel([1.0, 1.0, 1.0])
    # )

    # f(x)= 1.0*g(x) + 1.0*g(x-1) + 1.0*g(x-2) + 1.8*f(x-1) - 0.9*f(x-2)

    input = np.random.randint(0, 100, 20).tolist()

    compiler = IIRCompiler(
        filter=[filter]
    )
    print("Cost before dilation:", compiler.cost(16))

    expected = np.array(compiler.run(input))

    # compiler.dilate(0, 2)

    # manual dilation
    # mask = [False] * 8
    # mask[4] = True
    # mask[6] = True
    # compiler.reformulate(0, mask)

    # compiler.dilate(0, 8)
    # compiler.delay(-1, 8)
    # compiler.delay(-1, 16)

    compiler.reformulate(0, '00011111')
    compiler.reformulate(1, '00000111111')
    compiler.reformulate(2, '00000000011111111')

    print(compiler.filter)
    print("Cost after dilation:", compiler.cost(16))

    actual = np.array(compiler.run(input))
    
    print(np.vstack((expected, actual)).T)

if __name__ == "__main__":
    main()
