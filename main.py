from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="mixed",
        folds=10,
        indices=[
            ["savi_0p5","evi_2p5_6_7p5_1"]
        ]
    )
    c.process()