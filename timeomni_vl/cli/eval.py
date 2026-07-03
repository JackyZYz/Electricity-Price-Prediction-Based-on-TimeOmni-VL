import argparse

import numpy as np

from timeomni_vl.tasks.generation import GenerationEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    pred = np.load(args.pred)
    ref = np.load(args.ref)

    evaluator = GenerationEvaluator()
    result = evaluator.evaluate(pred, ref)
    print(result)


if __name__ == "__main__":
    main()
