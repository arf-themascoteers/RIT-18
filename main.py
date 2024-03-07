from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="mixed",
        folds=10,
        algorithms=[
            "ann_normal_savi",
            "ann_learnable_savi",
            "ann_learnable_simple_savi"
        ]
    )
    c.process()