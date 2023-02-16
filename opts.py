import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', default='Stress_dataset.zip',
                        type=str, help='Stress_dataset.zip Path')
    parser.add_argument('--survey_path', default='SurveyResults.xlsx',
                        type=str, help='SurveyResults.xlsx Path')
    parser.add_argument('--save_file', default='labeled_data.pkl',
                        type=str, help='Specify the path to save labeled data')
    parser.add_argument('--cpu_count', default=15, type=int,
                        help='Specify number of cores to be used')

    args = parser.parse_args()

    return args
