from typing import overload

from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.transform import *


def get_optimizing_schedule(max_bits: int) -> Transform:
    return Seq(
        Preorder(Try(Eliminate2DKernels())),
        SaturatePreorder(ConstantFold),
        Optional(Preorder(Try(Factorize()))),
        SaturatePreorder(ConstantFold),
        Optional(Preorder(Try(FuseRecurse()))),
        SaturatePreorder(ConstantFold),
        Optional(
            Seq(
                SaturatePreorderUpTo(Dilate(max_bits // 2)),
                SaturatePreorder(ConstantFold),
                Any(Dilate(), Delay()),
            )
        ),
        AnnotateLanes(max_bits),
        PushDownConvertLanes(),
    )


@overload
def get_candidate_schedules(expr: SignalExpr, max_bits: int) -> list[SignalExpr]: ...


@overload
def get_candidate_schedules(
    expr: SignalExpr2D, max_bits: int
) -> list[SignalExpr2D]: ...


def get_candidate_schedules(expr, max_bits):
    return get_optimizing_schedule(max_bits).apply_generic(expr)
