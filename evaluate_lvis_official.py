from lvis import LVISEval
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_anno_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()

    ANNOTATION_PATH = args.coco_anno_path
    RESULT_PATH = os.path.join(args.result_dir,"bbox.json")
    ANN_TYPE = 'bbox'

    lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH, ANN_TYPE)
    lvis_eval.run()
    lvis_eval.print_results()