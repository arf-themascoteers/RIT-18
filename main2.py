from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="mixed2",
        folds=10,
        algorithms=[
            "ann_learnable_simple_savi"
        ]
    )
    c.process()