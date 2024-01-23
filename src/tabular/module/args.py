import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="../../data",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data_3PL.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--model", default="lgbm", type=str, help="model name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data_3PL.csv", type=str, help="test file name"
    )
    parser.add_argument(
        "--fold", default=5, type=int, help="K-fold"
    )
    args = parser.parse_args()

    return args