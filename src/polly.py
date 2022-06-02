from cProfile import label
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding
from config import Config
from elg_service import predict_batch
import matplotlib.pyplot as plt


class BiasItem:
    def __init__(self, text: str = "", score_cultural: float = 0, score_socioeconomic: float = 0, party: str = "none",
                 id: str = ""):
        self.id: str = id
        self.party: str = party
        self.score_cultural: float = score_cultural
        self.score_socioeconomic: float = score_socioeconomic
        self.text: str = text

    @classmethod
    def from_json(cls, json_dict: dict = None):
        return BiasItem(**json_dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__) + "\n"


class BiasDataset(Dataset):
    def __getitem__(self, idx: int) -> BatchEncoding:
        with open(Config.DATASET_BIAS_PATH) as f:
            for i, line in enumerate(f.readlines()):
                if i == idx:
                    bi: BiasItem = BiasItem.from_json(json_dict=json.loads(line))
                    encodings: BatchEncoding
                    # TODO: tokenize text, add label
                    return encodings


def build_dataset():
    polly: pd.ExcelFile = pd.ExcelFile(Config.DATASET_POLLY_PATH)
    by_party_df: pd.DataFrame = pd.read_excel(polly, "by_party")
    by_column: str = "By"
    tweet_column: str = "Tweet"
    batch: list[BiasItem] = []
    all_rows = list(by_party_df.iterrows())
    with open(Config.DATASET_POLLY_CLASSIFIED_PATH, "a+") as f:
        for idx, row in tqdm(all_rows):
            text: str = row[tweet_column]
            batch.append(BiasItem(text=text, party=row[by_column], id=str(int(row["ID"]))))
            if len(batch) == 2 or idx == len(all_rows) - 1:
                socioeconomic, cultural = predict_batch(texts=[x.text for x in batch])
                for i in range(len(batch)):
                    batch[i].score_cultural = float(cultural[i])
                    batch[i].score_socioeconomic = float(socioeconomic[i])
                f.write("".join([x.to_json() for x in batch]))
                batch = []


def plot_dataset():
    with open(Config.DATASET_POLLY_CLASSIFIED_PATH) as f:
        bis: list[BiasItem] = []
        for line in f.readlines():
            if line:
                bi: BiasItem = BiasItem.from_json(json.loads(line[:-1]))
                bis.append(bi)
        bis_avg: list[BiasItem] = []
        parties: list[str] = ["AfD", "FDP", "CDU", "CSU", "Die Linke", "SPD", "Die Grünen"]
        for party in parties:
            party_bis: list[BiasItem] = [x for x in bis if x.party == party]
            cultural: float = np.mean([x.score_cultural for x in party_bis])
            socioeconomic: float = np.mean([x.score_socioeconomic for x in party_bis])
            bis_avg.append(BiasItem(score_cultural=cultural, score_socioeconomic=socioeconomic, party=party))
        color_dict: dict[str, str] = dict()
        party_colors: list[str] = ["b", "y", "k", "c", "m", "r", "g"]
        for i in range(len(parties)):
            color_dict[parties[i]] = party_colors[i]
        x = [x.score_cultural for x in bis_avg]
        y = [x.score_socioeconomic for x in bis_avg]
        c = [color_dict[x.party] for x in bis_avg]
        plt.plot([.5,.5],[0,1], linewidth=4, color='black')
        plt.plot([0,1],[.5,.5], linewidth=4, color='black')
        for i in range(len(x)):
            plt.scatter(x=x[i], y=y[i], c=c[i],label=bis_avg[i].party)  # , s=10
        plt.legend()
        plt.grid(True)
        plt.xlabel("Cultural Score from 0 (open) to 1 (conservative)")
        plt.ylabel("Socioeconomic Score from 0 (socialist) to 1 (liberal)")
        # plt.title("Cultural and Socioeconomic Scores of German Political Parties")
        plt.savefig("party_scores_cultural_socioeconomic.png", dpi=600)
        plt.show()


# build_dataset()
# plot_dataset()
