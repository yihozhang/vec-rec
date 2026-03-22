# f(x) = f(x-1) * 1.8 + f(x-2) * -0.9 + g(x)

from vecrec.expr import *
from vecrec.transform import *
from vecrec.codegen import *
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType


def main() -> None:
    with with_types(Type.TropMax, ElementType.I64):
        program = Recurse(TIKernel([Type.TropMax.zero(), 2, 5]), Var("g"))
    
    exprs = get_candidate_schedules(program, max_bits=512, debug=True)
    results = generate_and_run_benchmark(CodeGen(), exprs)
    print(results)
    


if __name__ == "__main__":
    main()
