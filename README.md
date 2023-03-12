# Data Labelling for Nurses Dataset
An example script to label the data for the paper
[A multimodal sensor dataset for continuous stress detection of nurses in a hospital](https://www.nature.com/articles/s41597-022-01361-y)

## Quickstart
- The dataset files (Stress_dataset.zip and SurveyResults.xlsx) can be found [here](https://zenodo.org/record/5514277#.Y-z_FnbMJPY)
- `git clone https://github.com/soon-jynn-chu/label_nurses_dataset.git`
- Install dependencies `pip install numpy pandas openpyxl`
- Edit the configurations in `opts.py` as needed
- Execute the file `python main.py`

### Note
- Entire process will take a few minutes (~5 minutes for 15 cores), specify more cores in `opts.py` to speed up
- Not including IBI and BVP signals
