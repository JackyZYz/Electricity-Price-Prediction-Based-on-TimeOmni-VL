import argparse

from timeomni_vl.config import ConfigManager
from timeomni_vl.data.pipeline import build_data_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = ConfigManager(args.config).get_data_config()
    pipeline = build_data_pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
