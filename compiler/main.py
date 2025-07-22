import random
from typing import List, Optional
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import lfilter
import argparse


class Signal:
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

    def copy(self) -> 'Signal':
        return Signal(self.data.copy())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        self.compress()
        data = " ".join([f"{x:.2f}" for x in self.data])
        return f"Signal([ {data} ])"

    def compress(self):
        """Remove trailing zeros from the signal."""
        while self.data[-1] == 0.0:
            self.data.pop()
    
    def num_terms(self) -> int:
        """Return the number of non-zero terms in the signal."""
        self.compress()
        return len(self.data) - self.data.count(0.)


class IIR:
    def __init__(self, a: Signal, b: Signal):
        """
        Initialize IIR compiler with feedback coefficients.
        
        Args:
            b: List of feedforward coefficients [b0, b1, b2, ...]
            a: List of feedback coefficients [a0, a1, a2, ...]
        
            Computes f(x) = b0*g(x) + b1*g(x-1) + ...
                          + a1*f(x-1) + a2*f(x-2) + ...
            a0 must be 1.0
        """
        assert a[0] == 1.0, "First coefficient must be 1.0"
        a.compress()
        b.compress()
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        self.a.compress()
        self.b.compress()
        return f"IIR(a={self.a}, b={self.b})"
    
    def reformulated(self, mask: List[bool]) -> Optional['IIR']:
        """
        Reformulate the IIR filter. The mask indicates which coefficients to keep,
        and the length of the mask indicates how farther back the reformulation can consider.
        For example, a mask with length 8 means we will consider coefficients up to 8 samples
        back (f(x), f(x-1), ..., f(x-7)).
        """
        self.a.compress()
        self.b.compress()
        mask = np.array(mask, dtype=bool)

        assert len(mask) >= len(self.a), "Mask must be longer than a coefficients"

        len_a = len(mask)
        len_b = len(mask) + len(self.b)

        num_rewrites = len(mask) - len(self.a) + 1
        A = np.zeros((len_a + len_b, num_rewrites))

        def make_rewrite(offset: int):
            rw_vec = [0.] * offset
            rw_vec.append(-1.0)
            rw_vec.extend(self.a.data[1:])
            assert len(rw_vec) <= len_a, "Rewrite vector exceeds reserved length for a coefficients"
            rw_vec.extend([0.] * (len_a - len(rw_vec)))

            rw_vec.extend([0.] * offset)
            rw_vec.extend(self.b.data)
            assert len(rw_vec) <= len_a + len_b, "Rewrite vector exceeds reserved length for b coefficients"
            rw_vec.extend([0.] * (len_a + len_b - len(rw_vec)))

            return np.array(rw_vec)

        for i in range(num_rewrites):
            A[:, i] = make_rewrite(i)

        assert not mask[0], "The first coefficient must not be kept"

        extended_mask = np.concatenate((mask, np.ones(len_b, dtype=bool)))
        A_zero_entries = A[~extended_mask, :]
        c = np.zeros(np.sum(~mask))
        c[0] = -1.

        sol, residual = lstsq(A_zero_entries, c)[0:2]
        if sum(residual) > 1e-6:
            return None  # No solution found
        else:
            coeffs = A @ sol

            b = Signal(coeffs[len_a:len_a + len_b].tolist())

            a = Signal(coeffs[:len_a].tolist())
            assert np.isclose(a[0], -1)
            a[0] = 1.0  # Ensure the first coefficient is 1.0

            return IIR(a, b)


    def delay(self, n: int) -> 'IIR':
        """Delay the IIR filter by n samples. This is not semantic-preserving."""
        # I'm doing a manual Gaussian elimination here, but it is possibel to do the same 
        # with reformulation.
        a = self.a.copy()
        b = self.b.copy()
        for i in range(1, n):
            for j in range(1, len(a)):
                a[j + i] += a[i] * self.a[j]
            for j in range(0, len(b)):
                b[j + i] += a[i] * self.b[j]
            a[i] = 0.
        return IIR(a, b)

    def dilate(self, lanes: int, step: int = 2) -> List['IIR']:
        """Expand the filter into a tower of dilating filters."""
        assert lanes > 0 and (lanes & (lanes - 1)) == 0, "Lanes must be a power of 2"

        if step > lanes:
            return [self]
        result_filters = [IIR(Signal([1.0]), Signal(self.b.data.copy()))]
        new_self = IIR(self.a.copy(), Signal([1.0]))
        order = 1
        dilating_filter = None
        while True:
            mask = [False] * (1 + order * step)
            for i in range(order):
                mask[(i + 1) * step] = True
            if len(mask) < len(self.a):
                mask.extend([False] * (len(self.a) - len(mask)))
            dilating_filter = new_self.reformulated(mask)
            if dilating_filter is None:
                order += 1
            else:
                break
        result_filters.extend(dilating_filter.dilate(lanes, step=step*2))

        return result_filters

    def run(self, x: List[float]) -> List[float]:
        """Run the filter on the input signal x."""
        y = []
        for i in range(len(x)):
            res = 0.
            for j in range(len(self.b)):
                if i - j >= 0:
                    res += self.b[j] * x[i - j]
            for j in range(1, len(self.a)):
                if i - j >= 0:
                    res += self.a[j] * y[i - j]
            y.append(res)
        return y
    
    def is_non_recursive(self) -> bool:
        """Check if the filter is non-recursive (no feedback)."""
        self.a.compress()
        self.b.compress()
        return len(self.a) == 1
    
    def feedback_delay(self) -> int:
        for i in range(1, len(self.a)):
            if self.a[i] != 0.:
                return i
        raise ValueError("The filter is not recursive")

class IIRCompiler:
    def __init__(self, filter: List[IIR]):
        self.filter = filter
    
    def _pos_to_idx(self, pos: int) -> int:
        return len(self.filter) - 1 - pos if pos < 0 else pos

    def delay(self, pos: int, n: int) -> None:
        """Delay the filter at position pos by n samples."""
        idx = self._pos_to_idx(pos)

        self.filter[idx] = self.filter[idx].delay(n)

    def dilate(self, pos: int, lanes: int) -> None:
        """Expand the filter into a tower of dilating filters."""
        idx = self._pos_to_idx(pos)

        filters = self.filter[idx].dilate(lanes)
        self.filter = self.filter[:idx] + filters + self.filter[idx + 1:]

    def cost(self, max_lanes: int) -> float:
        cost = 0.
        for f in self.filter:
            cost += f.b.num_terms() / max_lanes
            if not f.is_non_recursive():
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
    np.set_printoptions(precision=2)
    # filter = IIR(
    #     b=Signal([0.1, 0.2, 0.3]),
    #     a=Signal([1.0, -0.5, 0.25])
    # )
    filter = IIR(
        b=Signal([1., 1., 1.]),
        a=Signal([1.0, 1.8, -0.9])
    )

    input = random.sample(range(100), 10)

    compiler = IIRCompiler(
        filter=[filter]
    )
    print("Cost before dilation:", compiler.cost(16))

    print(np.array(compiler.run(input)))

    # manual dilation
    # mask = [False] * 8
    # mask[2] = True
    # mask[4] = True
    # compiler.filter[0] = compiler.filter[0].reformulated(mask)

    compiler.dilate(0, 8)
    print(compiler.filter)
    print("Cost after dilation:", compiler.cost(16))



    # compiler.delay(0, 8)
    print(np.array(compiler.run(input)))

if __name__ == "__main__":
    main()
