from typing import overload

from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.transform import *
from vecrec.transform.core import DilateLTVTrop, DilateLTVArithVec, DilateLTVTropVec, DilateLTVHalfPointsVec


def get_optimizing_schedule(max_bits: int, debug: bool = False) -> Transform:
    # fmt: off
    return Seq(
        Any(
            Seq(
                Preorder(Try(Eliminate2DKernels())),
                SaturatePreorder(ConstantFold),
            ),
            Seq(
                Preorder(Try(Factorize2D())),
                SaturatePreorder(ConstantFold),
                SaturatePreorder(InlinePointwiseOps()),

                SaturatePreorder(
                    Seq(
                        PushDownAndFuseConvolve(),
                        Try(EliminateDegenerate2Dkernels()))
                ),
                SaturatePreorder(ConstantFold),

                # SaturatePreorder(DilateTVWithSingleOddOrder()),
                # SaturatePreorder(ConstantFold),

                Preorder(Try(DilateLTVHalfPoints(max_bits))),
                SaturatePreorder(ConstantFold),
                Preorder(Try(DilateLTVHalfPointsVec(max_bits))),
                SaturatePreorder(ConstantFold),

                Preorder(Try(DilateLTVTrop(max_bits))),
                SaturatePreorder(ConstantFold),
                Preorder(Try(DilateLTVTropVec(max_bits))),
                SaturatePreorder(ConstantFold),

                Preorder(Try(DilateLTVArith(max_bits))),
                SaturatePreorder(ConstantFold),
                Preorder(Try(DilateLTVArithVec(max_bits))),
                SaturatePreorder(ConstantFold),
                
                Preorder(Try(Eliminate2DKernels())),
                SaturatePreorder(ConstantFold),

                Optional(Preorder(Try(Factorize()))),
                SaturatePreorder(ConstantFold),

                Optional(Preorder(Try(FuseRecurse()))),
                SaturatePreorder(ConstantFold),

                Optional(
                    Seq(
                        SaturatePreorderUpTo(Any(Dilate(max_bits // 2), DilateTrop(max_bits // 2))),
                        SaturatePreorder(ConstantFold),
                        Preorder(Try(Any(Dilate(), Delay(), DilateTrop()))),
                        SaturatePreorder(ConstantFold),
                    )
                ),
            )
        ),
        # DebugPrint() if debug else Noop(),
        # PrintSizes(),

        InferNRows(),
        AnnotateLanes(max_bits),
        PushDownConvertLanes(),
    )


@overload
def get_candidate_schedules(expr: SignalExpr, max_bits: int, debug: bool) -> list[SignalExpr]: ...
@overload
def get_candidate_schedules(expr: SignalExpr2D, max_bits: int, debug: bool) -> list[SignalExpr2D]: ...
@overload
def get_candidate_schedules(expr: SignalExpr, max_bits: int) -> list[SignalExpr]: ...
@overload
def get_candidate_schedules(expr: SignalExpr2D, max_bits: int) -> list[SignalExpr2D]: ...


def get_candidate_schedules(expr, max_bits, debug=False):
    return get_optimizing_schedule(max_bits, debug=debug).apply_generic(expr)
