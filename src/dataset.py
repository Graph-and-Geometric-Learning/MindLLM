import PIL.Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import random
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
from datasets import load_dataset
from lightning import LightningDataModule
from glob import glob
import PIL
from transformers import AutoTokenizer
from torchvision import transforms
import os
import json
import h5py
import numpy as np

from src.utils.chat import apply_chat_template

from enum import Enum


class DatasetSource(Enum):
    COCO_CAPTION = "coco-caption"
    PARAGRAPH_CAPTION = "paragraph-caption"
    COCO_QA = "coco-qa"
    VISUAL_GENOME = "visual-genome"
    VQA_V2 = "vqa-v2"
    OK_VQA = "ok-vqa"
    ST_VQA = "st-vqa"
    TALLY_QA = "tally-qa"
    VQA_E = "vqa-e"
    VSR = "vsr"
    A_OKVQA = "a-okvqa"
    FSVQA = "fsvqa"
    VISDIAL = "visdial"
    LLAVA = "llava"
    LVIS = "lvis"
    TDIUC = "tdiuc"

    GQA = "gqa"

    COCO_CAPTION_PREVIOUS = "coco-caption-previous"


IMAGE_SOURCES = list(
    map(
        lambda x: x.value,
        [
            DatasetSource.COCO_CAPTION,
            DatasetSource.PARAGRAPH_CAPTION,
            DatasetSource.COCO_QA,
            DatasetSource.VISUAL_GENOME,
            DatasetSource.VQA_V2,
            DatasetSource.OK_VQA,
            DatasetSource.ST_VQA,
            DatasetSource.TALLY_QA,
            DatasetSource.VQA_E,
            DatasetSource.VSR,
            DatasetSource.A_OKVQA,
            DatasetSource.FSVQA,
            DatasetSource.VISDIAL,
            DatasetSource.LLAVA,
            DatasetSource.LVIS,
            DatasetSource.GQA,
        ],
    )
)
assert len(IMAGE_SOURCES) == 16

IGNORE_INDEX = -100

info = pd.read_csv(os.path.join("data", "nsd_stim_info_merged.csv"))
nsd_to_image_id = (
    info[["Unnamed: 0", "cocoId"]]
    .set_index("Unnamed: 0", drop=True)
    .to_dict()["cocoId"]
)


def load_aokvqa(aokvqa_dir, split, version="v1p0"):
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset = json.load(
        open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json"))
    )
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def load_coco_image(image_id, coco_dir):
    for split in ["train", "val", "test"]:
        path = get_coco_path(split, image_id, coco_dir)
        if os.path.exists(path):
            return PIL.Image.open(path)
    raise FileNotFoundError


FMRI_KEY = "fMRI"
FMRI_TOKEN = "<" + FMRI_KEY + ">"


def post_process_prompt(prompt):
    prompt = prompt.replace("image", FMRI_KEY)
    if random.random() > 1.0:  # FIXME
        prompt = prompt + "\n" + FMRI_TOKEN
    else:
        prompt = FMRI_TOKEN + "\n" + prompt
    return prompt


def create_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert tokenizer.pad_token_id is not None
    tokenizer.add_tokens(FMRI_TOKEN)
    # assert tokenizer.tokenize(FMRI_TOKEN) == [FMRI_TOKEN]
    # tokenizer.padding_side = 'left'
    return tokenizer


class InstructionDatasetLoader:
    def __init__(self, file, whole_brain: bool = False):
        self.file = file
        self.whole_brain = whole_brain
        self.tdiuc_filter = None

    def from_dataset(self, source, constructor):
        cacheable = True
        cache_path = os.path.join("data", f"cache", f"{source}.json")
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path), exist_ok=False)
        if cacheable and os.path.exists(cache_path):
            dataset = json.load(open(cache_path))
        else:
            dataset = constructor()
            if cacheable:
                json.dump(dataset, open(cache_path, "w"))
        logging.info(f"Loaded {len(dataset)} samples from {source}")

        if self.tdiuc_filter is not None:
            dataset = [
                sample
                for sample in dataset
                if sample["answers"][0] == self.tdiuc_filter
            ]
            assert len(dataset) > 0, f"maybe wrong tdiuc filter {self.tdiuc_filter}"

        def stats_each_split(dataset):
            length = {"train": 0, "val": 0, "test": 0}
            for sample in dataset:
                length[sample["split"]] += 1
            logging.info(
                f"Reading train {length['train']}, val {length['val']}, test {length['test']} from {source}"
            )

        def sanity_check(dataset):
            for sample in dataset:
                assert "coco_id" in sample or "sample_id" in sample
                if "coco_id" in sample:
                    assert isinstance(sample["coco_id"], int)
                assert "from" in sample
                assert "chat" in sample
                chat = sample["chat"]

                prev_role = "assistant"
                for message in chat:
                    role = message["role"]
                    content = message["content"]
                    if prev_role == "assistant":
                        assert role == "user"
                    elif prev_role == "user":
                        assert role == "assistant"
                    else:
                        raise
                    prev_role = role
                    assert isinstance(content, str)

        sanity_check(dataset)
        stats_each_split(dataset)

        return dataset

    def from_coco_caption(self):
        dataset = []
        for split in ["train", "val"]:
            annotations = json.load(
                open(f"data/coco/annotations/captions_{split}2017.json")
            )["annotations"]
            for sample in annotations:
                coco_id = int(sample["image_id"])
                dataset.append(
                    {
                        "coco_id": coco_id,
                        "from": DatasetSource.COCO_CAPTION.value,
                        "split": split,
                        "unique_key": coco_id,
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    f"Please describe the {FMRI_KEY} as simply as possible."
                                ),
                            },
                            {"role": "assistant", "content": sample["caption"]},
                        ],
                    }
                )
        answers = defaultdict(list)
        for sample in dataset:
            answers[sample["coco_id"]].append(sample["chat"][1]["content"])
        for sample in dataset:
            sample["answers"] = answers[sample["coco_id"]]

        return dataset

    def from_coco_caption_previous(self):
        dataset = []
        annotations_ = (
            json.load(open(f"data/coco/annotations/captions_train2017.json"))[
                "annotations"
            ]
            + json.load(open(f"data/coco/annotations/captions_val2017.json"))[
                "annotations"
            ]
        )
        annotations = defaultdict(list)
        for ann in annotations_:
            annotations[ann["image_id"]].append(ann["caption"])
        unique_key = 0
        for split in ["train", "val", "test"]:
            if split not in self.file:
                continue
            for subj, file_sub in self.file[split].items():
                for key in file_sub:
                    data = file_sub[key]
                    prev_coco_id = int(data["prev_coco_id"][()])
                    if prev_coco_id == -1:
                        continue
                    captions = annotations[prev_coco_id]
                    for caption in captions:
                        dataset.append(
                            {
                                "sample_id": subj + "-" + split + "-" + key,
                                "from": DatasetSource.COCO_CAPTION_PREVIOUS.value,
                                "split": split,
                                "unique_key": unique_key,
                                "answers": captions,
                                "chat": [
                                    {
                                        "role": "user",
                                        "content": post_process_prompt(
                                            f"Please describe the image the subject saw previously."
                                        ),
                                    },
                                    {"role": "assistant", "content": caption},
                                ],
                            }
                        )
                    unique_key += 1

        assert "coco_id" not in dataset[0]

        return dataset

    def from_coco_qa(self):
        instruction = "\nAnswer the question with a short phrase."
        dataset = []
        unique_key = 0
        for split in ["train", "test"]:
            questions = (
                open(f"data/coco-qa/{split}/questions.txt")
                .read()
                .rstrip()
                .rstrip("\n")
                .split("\n")
            )
            answers = (
                open(f"data/coco-qa/{split}/answers.txt")
                .read()
                .rstrip()
                .rstrip("\n")
                .split("\n")
            )
            img_ids = (
                open(f"data/coco-qa/{split}/img_ids.txt")
                .read()
                .rstrip()
                .rstrip("\n")
                .split("\n")
            )
            assert len(questions) == len(answers) == len(img_ids)
            for question, answer, img_id in zip(questions, answers, img_ids):
                prompt = post_process_prompt(question + instruction)
                dataset.append(
                    {
                        "coco_id": int(img_id),
                        "from": DatasetSource.COCO_QA.value,
                        "split": split,
                        "unique_key": unique_key,
                        "answers": answer,
                        "chat": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
                unique_key += 1
        return dataset

    def from_image_paragraph_captioning(self):
        instruction = f"Describe the {FMRI_KEY} in one paragraph."
        dataset = []
        meta = {
            item["image_id"]: item["coco_id"]
            for item in json.load(open("data/visual-genome/image_data.json"))
        }

        samples = json.load(open("./data/paragraph-captioning/paragraphs_v1.json"))
        for split in ["train", "val", "test"]:
            split_ids = set(
                json.load(open(f"./data/paragraph-captioning/{split}_split.json"))
            )
            for sample in samples:
                if sample["image_id"] not in split_ids:
                    continue
                coco_id = meta[sample["image_id"]]
                if coco_id is None:
                    continue
                dataset.append(
                    {
                        "coco_id": coco_id,
                        "from": DatasetSource.PARAGRAPH_CAPTION.value,
                        "split": split,
                        "unique_key": coco_id,
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(instruction),
                            },
                            {"role": "assistant", "content": sample["paragraph"]},
                        ],
                    }
                )
        answers = defaultdict(list)
        for sample in dataset:
            answers[sample["coco_id"]].append(sample["chat"][1]["content"])
        for sample in dataset:
            sample["answers"] = answers[sample["coco_id"]]
        return dataset

    def from_visual_genome_qa(self):
        instruction = "\nAnswer the question with a short phrase."
        meta = {
            item["image_id"]: item["coco_id"]
            for item in json.load(open("data/visual-genome/image_data.json"))
        }
        all_qas = json.load(open("data/visual-genome/question_answers.json"))
        dataset = []
        split_ids = {
            k: set(v)
            for k, v in json.load(
                open("data/visual-genome/densecap_splits.json")
            ).items()
        }
        unique_key = 0
        for split in ["train", "val", "test"]:
            for qas in all_qas:
                for qa in qas["qas"]:
                    if qa["image_id"] not in split_ids[split]:
                        continue
                    coco_id = meta[qa["image_id"]]
                    if coco_id is None:
                        continue
                    dataset.append(
                        {
                            "coco_id": coco_id,
                            "from": DatasetSource.VISUAL_GENOME.value,
                            "split": split,
                            "unique_key": unique_key,
                            "answers": qa["answer"],
                            "chat": [
                                {
                                    "role": "user",
                                    "content": post_process_prompt(
                                        qa["question"] + instruction
                                    ),
                                },
                                {"role": "assistant", "content": qa["answer"]},
                            ],
                        }
                    )
                    unique_key += 1
        return dataset

    def from_vqav2(self):
        instruction = "\nAnswer the question with a short phrase."
        dataset = []
        for split in ["train", "val"]:
            questions = {
                question.pop("question_id"): question
                for question in json.load(
                    open(f"data/VQAv2/v2_OpenEnded_mscoco_{split}2014_questions.json")
                )["questions"]
            }
            annotations = json.load(
                open(f"data/VQAv2/v2_mscoco_{split}2014_annotations.json")
            )["annotations"]
            for annotation in annotations:
                question = questions[annotation["question_id"]]["question"]
                answers = annotation["answers"]
                answers = [answer["answer"] for answer in answers]
                for answer in answers:
                    dataset.append(
                        {
                            "coco_id": annotation["image_id"],
                            "from": DatasetSource.VQA_V2.value,
                            "question_id": annotation["question_id"],
                            "unique_key": annotation["question_id"],
                            "answers": answers,
                            "split": split,
                            "chat": [
                                {
                                    "role": "user",
                                    "content": post_process_prompt(
                                        question + instruction
                                    ),
                                },
                                {"role": "assistant", "content": answer},
                            ],
                        }
                    )
        return dataset

    def from_okvqa(self):
        instruction = "\nAnswer the question with a short phrase."
        dataset = []
        for split in ["train", "val"]:
            questions = {
                question.pop("question_id"): question
                for question in json.load(
                    open(f"data/okvqa/OpenEnded_mscoco_{split}2014_questions.json")
                )["questions"]
            }
            annotations = json.load(
                open(f"data/okvqa/mscoco_{split}2014_annotations.json")
            )["annotations"]
            for annotation in annotations:
                question = questions[annotation["question_id"]]["question"]
                answers = annotation["answers"]
                answers = [answer["answer"] for answer in answers]
                for answer in answers:
                    dataset.append(
                        {
                            "coco_id": annotation["image_id"],
                            "question_id": annotation["question_id"],
                            "unique_key": annotation["question_id"],
                            "from": DatasetSource.OK_VQA.value,
                            "split": split,
                            "answers": answers,
                            "chat": [
                                {
                                    "role": "user",
                                    "content": post_process_prompt(
                                        question + instruction
                                    ),
                                },
                                {"role": "assistant", "content": answer},
                            ],
                        }
                    )
        return dataset

    def from_st_vqa(self):
        instruction = "\nAnswer the question with a short phrase."
        dataset = []
        unique_key = 0
        for split in ["train"]:
            for task_i in range(1, 4):
                samples = json.load(open(f"data/st-vqa/{split}_task_{task_i}.json"))[
                    "data"
                ]
                for sample in samples:
                    if "coco" not in sample["dataset"].lower():
                        continue
                    coco_id = int(sample["file_name"].split(".")[0].split("_")[-1])
                    answers = sample["answers"]
                    for answer in answers:
                        dataset.append(
                            {
                                "from": DatasetSource.ST_VQA.value,
                                "coco_id": coco_id,
                                "split": split,
                                "unique_key": unique_key,
                                "answers": answers,
                                "chat": [
                                    {
                                        "role": "user",
                                        "content": post_process_prompt(
                                            sample["question"] + instruction
                                        ),
                                    },
                                    {"role": "assistant", "content": answer},
                                ],
                            }
                        )
                    unique_key += 1
        return dataset

    def from_tallyqa(self):
        instruction = "\nAnswer the question with a number."
        meta = {
            item["image_id"]: item["coco_id"]
            for item in json.load(open("data/visual-genome/image_data.json"))
        }
        dataset = []
        unique_key = 0
        for split in ["train", "test"]:
            samples = json.load(open(f"data/tallyqa/{split}.json"))
            for sample in samples:
                if sample["data_source"] == "imported_genome":
                    assert "VG" in sample["image"]
                    image_id = sample["image_id"]
                    assert str(image_id).startswith("9")
                    image_id = image_id - 90000000
                    coco_id = meta[image_id]
                    if coco_id is None:
                        continue
                elif "COCO" in sample["image"]:
                    coco_id = int(sample["image"].split(".")[0].split("_")[-1])
                    assert coco_id == sample["image_id"]
                else:
                    continue
                dataset.append(
                    {
                        "from": DatasetSource.TALLY_QA.value,
                        "coco_id": coco_id,
                        "split": split,
                        "unique_key": unique_key,
                        "answers": str(sample["answer"]),
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    sample["question"] + instruction
                                ),
                            },
                            {"role": "assistant", "content": str(sample["answer"])},
                        ],
                    }
                )
                unique_key += 1
        return dataset

    def from_vqa_e(self):
        instruction = (
            "\nAnswer with a short phrase and provide explanation for your answer."
        )
        dataset = []
        unique_key = 0
        for split in ["train", "val"]:
            samples = json.load(open(f"data/VQA-E/VQA-E_{split}_set.json"))
            for sample in samples:
                question = sample["question"]
                explanation = sample["explanation"][0]
                for answer in sample["answers"]:
                    response = (
                        answer.rstrip(".")
                        + ". Here is the explanation:\n"
                        + explanation
                    )  # FIXME: add the
                    dataset.append(
                        {
                            "from": DatasetSource.VQA_E.value,
                            "coco_id": sample["img_id"],
                            "unique_key": unique_key,
                            "split": split,
                            "answers": (sample["answers"], explanation),
                            "chat": [
                                {
                                    "role": "user",
                                    "content": post_process_prompt(
                                        question + instruction
                                    ),
                                },
                                {"role": "assistant", "content": response},
                            ],
                        }
                    )
                unique_key += 1
        return dataset

    def from_vsr(self):
        instruction = "\nPlease answer yes or no about whether the statement about the image is true."
        dataset = []
        data_files = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}
        samples = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
        unique_key = 0
        for split in ["train", "val", "test"]:
            for sample in samples[split]:
                if sample["label"] == 0:
                    label = "No."
                elif sample["label"] == 1:
                    label = "Yes."
                else:
                    raise
                dataset.append(
                    {
                        "from": DatasetSource.VSR.value,
                        "coco_id": int(sample["image"].split(".")[0]),
                        "split": split,
                        "unique_key": unique_key,
                        "answers": label,
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    sample["caption"] + instruction
                                ),
                            },
                            {"role": "assistant", "content": label},
                        ],
                    }
                )
                unique_key += 1

        return dataset

    def from_aokvqa(self):
        instruction = "\nAnswer with option's letter from the given choices and provide explanation for your choice."
        dataset = []
        unique_key = 0
        for split in ["train", "val"]:
            samples = load_aokvqa("./data/aokvqa", split)
            for sample in samples:
                prompt = sample["question"]
                prompt += "\nMultiple Choices:\n"
                for i, choice in enumerate(sample["choices"]):
                    letter = chr(ord("A") + i)
                    prompt += letter + ". " + choice + "\n"
                prompt += instruction
                answer = chr(ord("A") + sample["correct_choice_idx"]) + "."
                dataset.append(
                    {
                        "from": DatasetSource.A_OKVQA.value,
                        "coco_id": sample["image_id"],
                        "split": split,
                        "unique_key": unique_key,
                        "answers": answer,
                        "chat": [
                            {"role": "user", "content": post_process_prompt(prompt)},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
                unique_key += 1
        return dataset

    def from_fsvqa(self):
        instruction = "\nAnswer the question in a full sentence."
        dataset = []
        unique_key = 0
        references = (
            json.load(open(f"data/VQA/mscoco_train2014_annotations.json"))[
                "annotations"
            ]
            + json.load(open(f"data/VQA/mscoco_val2014_annotations.json"))[
                "annotations"
            ]
        )
        references = {item["question_id"]: item for item in references}

        for split in ["train", "val"]:
            questions = json.load(
                open(f"data/fsvqa/fsvqa_original_{split}_questions.json")
            )["questions"]
            questions = {
                question.pop("question_id"): question for question in questions
            }
            annotations = json.load(
                open(f"data/fsvqa/fsvqa_original_{split}_annotations.json")
            )["annotations"]

            for annotation in annotations:
                question = questions[annotation["question_id"]]
                assert question["img_id"] == annotation["img_id"]
                assert len(annotation["answers"]) == 1
                reference = references[annotation["question_id"]]
                short_answer = reference["multiple_choice_answer"]
                all_answers = [answer["answer"] for answer in reference["answers"]]
                full_answer = annotation["answers"][0]["answer"]
                dataset.append(
                    {
                        "from": DatasetSource.FSVQA.value,
                        "coco_id": annotation["img_id"],
                        "split": split,
                        "unique_key": unique_key,
                        "answers": (short_answer, full_answer, all_answers),
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    question["question"] + instruction
                                ),
                            },
                            {"role": "assistant", "content": full_answer},
                        ],
                    }
                )
                unique_key += 1
        return dataset

    def from_visdial(self):
        dataset = []
        unique_key = 0
        for split in ["train", "val"]:
            data = json.load(open(f"data/visdial/visdial_1.0_{split}.json"))
            questions = data["data"]["questions"]
            answers = data["data"]["answers"]
            dialogs = data["data"]["dialogs"]
            for dialog in dialogs:
                coco_id = dialog["image_id"]
                chat = []
                for i, qa in enumerate(dialog["dialog"]):
                    if i == 0:
                        chat.append(
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    questions[qa["question"]]
                                ),
                            }
                        )
                    else:
                        chat.append(
                            {"role": "user", "content": questions[qa["question"]]}
                        )
                    chat.append({"role": "assistant", "content": answers[qa["answer"]]})
                dataset.append(
                    {
                        "from": DatasetSource.VISDIAL.value,
                        "split": "train",
                        "unique_key": unique_key,
                        "coco_id": coco_id,
                        "chat": chat,
                    }
                )
                unique_key += 1

        return dataset

    def from_llava(self):
        def transform(sample, unique_key):
            return {
                "coco_id": int(sample["id"]),
                "from": DatasetSource.LLAVA.value,
                "unique_key": unique_key,
                "split": "train",
                "chat": [
                    {
                        "role": (
                            "user"
                            if s["from"] == "human"
                            else ("assistant" if s["from"] == "gpt" else None)
                        ),
                        "content": s["value"]
                        .replace("<image>", FMRI_TOKEN)
                        .replace("image", FMRI_KEY),
                    }
                    for s in sample["conversations"]
                ],
            }

        dataset = [
            transform(sample, i)
            for i, sample in enumerate(
                json.load(open("./data/llava_instruct_150k.json"))
            )
        ]
        return dataset

    def from_lvis(self):
        def transform(sample, unique_key):
            assert "coco" in sample["id"]
            coco_id = int(sample["id"].split("/")[-1])
            return {
                "coco_id": coco_id,
                "from": DatasetSource.LVIS.value,
                "split": "train",
                "unique_key": unique_key,
                "chat": [
                    {
                        "role": (
                            "user"
                            if s["from"] == "human"
                            else ("assistant" if s["from"] == "gpt" else None)
                        ),
                        "content": s["value"]
                        .replace("<image>", FMRI_TOKEN)
                        .replace("image", FMRI_KEY),
                    }
                    for s in sample["conversations"]
                ],
            }

        dataset = [
            transform(sample, i)
            for i, sample in enumerate(
                json.load(open("./data/lvis_instruct4v_220k.json"))
            )
        ]
        return dataset

    def from_tdiuc(self):
        instruction = "\nAnswer the question with a short phrase."
        dataset = []
        questions = json.load(
            open("data/tdiuc/OpenEnded_mscoco_train2014_questions.json")
        )["questions"]
        questions = {question.pop("question_id"): question for question in questions}
        annotations = json.load(open("data/tdiuc/mscoco_train2014_annotations.json"))[
            "annotations"
        ]
        for annotation in annotations:
            annotation["image_id"]
            question = questions[annotation["question_id"]]
            answers = [answer["answer"] for answer in annotation["answers"]]
            assert len(answers) == 1
            for answer in answers:
                dataset.append(
                    {
                        "coco_id": annotation["image_id"],
                        "from": DatasetSource.TDIUC.value,
                        "unique_key": annotation["question_id"],
                        "answers": (annotation["question_type"], answer),
                        "split": "train",
                        "chat": [
                            {
                                "role": "user",
                                "content": post_process_prompt(
                                    question["question"] + instruction
                                ),
                            },
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
        return dataset


class InstructionDatasetBuilder:
    def __init__(self, file, subjects=[1], datasets=None, whole_brain=False):
        self.file = file
        self.subjects = subjects
        self.whole_brain = whole_brain
        self.instruction_dataloader = InstructionDatasetLoader(
            file, whole_brain=whole_brain
        )
        instruction_dataloader = self.instruction_dataloader
        if datasets is None:
            sources = {
                DatasetSource.COCO_CAPTION_PREVIOUS.value: instruction_dataloader.from_coco_caption_previous,
                DatasetSource.COCO_CAPTION.value: instruction_dataloader.from_coco_caption,
                DatasetSource.PARAGRAPH_CAPTION.value: instruction_dataloader.from_image_paragraph_captioning,
                DatasetSource.COCO_QA.value: instruction_dataloader.from_coco_qa,
                DatasetSource.VISUAL_GENOME.value: instruction_dataloader.from_visual_genome_qa,
                DatasetSource.VQA_V2.value: instruction_dataloader.from_vqav2,
                DatasetSource.OK_VQA.value: instruction_dataloader.from_okvqa,
                DatasetSource.ST_VQA.value: instruction_dataloader.from_st_vqa,
                DatasetSource.TALLY_QA.value: instruction_dataloader.from_tallyqa,
                DatasetSource.VQA_E.value: instruction_dataloader.from_vqa_e,
                DatasetSource.VSR.value: instruction_dataloader.from_vsr,
                DatasetSource.A_OKVQA.value: instruction_dataloader.from_aokvqa,
                DatasetSource.FSVQA.value: instruction_dataloader.from_fsvqa,
                DatasetSource.VISDIAL.value: instruction_dataloader.from_visdial,
                DatasetSource.LLAVA.value: instruction_dataloader.from_llava,
                DatasetSource.LVIS.value: instruction_dataloader.from_lvis,
                # DatasetSource.TDIUC.value: instruction_dataloader.from_tdiuc,
            }
            self.datasets = sources

        else:
            self.datasets = datasets

    def load(self):
        for key, value in self.datasets.items():
            self.datasets[key] = self.instruction_dataloader.from_dataset(key, value)
        return self

    def new(self, **new_kwargs):
        kwargs = {
            "file": self.file,
            "subjects": self.subjects,
            "datasets": self.datasets,
            "whole_brain": self.whole_brain,
        }
        kwargs.update(new_kwargs)
        return InstructionDatasetBuilder(**kwargs)

    def filter(self, source=None):
        if source is not None:
            datasets = {source: self.datasets[source]}
            return self.new(datasets=datasets)
        return self

    def exclude(self, source):
        if source is None:
            return self
        assert source in self.datasets
        datasets = {k: v for k, v in self.datasets.items() if k != source}
        return self.new(datasets=datasets)

    def unique(self):
        datasets_uniqued = {}
        for key, dataset in self.datasets.items():
            existing = set()
            uniqued = []
            for sample in dataset:
                very_unique_key = (
                    sample["from"]
                    + "_"
                    + sample["split"]
                    + "_"
                    + str(sample["unique_key"])
                )
                if very_unique_key in existing:
                    continue
                existing.add(very_unique_key)
                uniqued.append(sample)
            assert len(uniqued) > 1
            datasets_uniqued[key] = uniqued
        return self.new(datasets=datasets_uniqued)

    def build(self):
        datasets = list(self.datasets.values())
        datasets = sum(datasets, [])
        logging.info(f"Total {len(datasets)}")

        datasets_by_coco = defaultdict(list)
        datasets_by_sample = defaultdict(list)
        for sample in datasets:
            if "coco_id" in sample:
                datasets_by_coco[sample["coco_id"]].append(sample)
            else:
                datasets_by_sample[sample["sample_id"]].append(sample)

        new_datasets_by_sample = {}
        for subj in self.subjects:
            for split in ["train", "val", "test"]:
                if len(self.file.get(split, [])) == 0:
                    continue
                for key, data in self.file[split][f"subject_{subj}"].items():
                    coco_id = int(data["coco_id"][()])
                    subj_key = f"subject_{subj}-{split}-{key}"
                    instructions = (
                        datasets_by_coco[coco_id] + datasets_by_sample[subj_key]
                    )
                    if len(instructions) > 0:
                        new_datasets_by_sample[subj_key] = Instructions(instructions)

        new_datasets_by_coco = {}
        for key, sample in datasets_by_coco.items():
            new_datasets_by_coco[key] = Instructions(sample)

        return InstructionDataset(new_datasets_by_sample, new_datasets_by_coco)


class Instructions:
    def __init__(self, instructions):
        self.instructions = instructions
        instructions_by_source = defaultdict(list)
        for instruction in instructions:
            instructions_by_source[instruction["from"]].append(instruction)
        self.instructions_by_source = instructions_by_source

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.instructions_by_source[key]
        elif isinstance(key, int):
            return self.instructions[key]
        else:
            raise TypeError(f"Invalid key {key}")

    def keys(self):
        return self.instructions_by_source.keys()


class InstructionDatasetBySubject:
    def __init__(self, instruction_datasets):
        self.instruction_datasets = instruction_datasets

    def __getitem__(self, key):
        return self.instruction_datasets[key]


class InstructionDataset:
    def __init__(self, datasets_by_sample, datasets_by_coco):
        self.datasets_by_sample = datasets_by_sample
        self.datasets_by_coco = datasets_by_coco

    def __getitem__(self, key):
        if key in self.datasets_by_sample:
            assert isinstance(key, str)
            return self.datasets_by_sample[key]
        elif key in self.datasets_by_coco:
            assert isinstance(key, int)
            return self.datasets_by_coco[key]
        else:
            raise

    def __contains__(self, key):
        return key in self.datasets_by_sample or key in self.datasets_by_coco

    def __len__(self):
        return sum([len(v) for v in self.datasets_by_sample.values()]) + sum(
            [len(v) for v in self.datasets_by_coco.values()]
        )


class DataCollator:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, batch):
        voxels = np.stack([b["voxels"] for b in batch])

        if self.tokenizer is None:
            chat = [b["chat"] for b in batch]
        else:
            chat = self.format_text_inputs(
                {
                    "input_ids": [b["chat"]["input_ids"] for b in batch],
                    "labels": [
                        b["chat"]["labels" if "labels" in b["chat"] else "input_ids"]
                        for b in batch
                    ],
                }
            )

        output = {
            "voxels": torch.from_numpy(voxels).float(),
            "chat": chat,
        }

        for prompt_key in ["prompt", "prompt2"]:
            if prompt_key in batch[0]:
                if self.tokenizer is None:
                    prompt = [b[prompt_key] for b in batch]
                else:
                    prompt = self.format_text_inputs(
                        {
                            "input_ids": [b[prompt_key]["input_ids"] for b in batch],
                            "labels": [
                                b[prompt_key][
                                    "labels" if "labels" in b["chat"] else "input_ids"
                                ]
                                for b in batch
                            ],
                        }
                    )
                # output[prompt_key]['response'] = [b[prompt_key]['response'] for b in batch]
                output[prompt_key] = prompt

        batch[0]["source"]
        output["source"] = [b["source"] for b in batch]
        output["subject"] = [b["subject"] for b in batch]
        output["sample_or_coco_id"] = [b["sample_or_coco_id"] for b in batch]
        output["answers"] = [b["answers"] for b in batch]
        output["question_id"] = [b["question_id"] for b in batch]
        # output['coco_id'] = [b['coco_id'] for b in batch]
        output["image"] = (
            torch.stack([b["image"] for b in batch]) if "image" in batch[0] else None
        )
        first_chats = [b["original_chat"]["chat"][0]["content"] for b in batch]
        if self.tokenizer is None:
            output["first_chat"] = first_chats
        else:
            output["first_chat"] = (
                self.tokenizer(
                    first_chats, return_tensors="pt", padding=True, truncation=True
                )
                if "original_chat" in batch[0]
                else None
            )
        return output

    def format_text_inputs(self, inputs):
        chat = self.tokenizer.pad(
            {
                "input_ids": inputs["input_ids"],
            },
            return_tensors="pt",
            padding=True,
        )
        labels = self.tokenizer.pad(
            {
                "input_ids": inputs["labels"],
            },
            return_tensors="pt",
            padding=True,
            return_attention_mask=False,
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        chat["labels"] = labels
        return chat


class MultiSourceBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Dataset,
        # sources: list,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        by="sample",
    ):
        self.data_source = data_source
        # self.sources = sources
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.by = by

        self._length = -1

    def __iter__(self):
        indices = self.data_source._get_keys(by=self.by)
        if self.shuffle:
            np.random.shuffle(indices)
        indices.sort(key=lambda x: x[0])  # sort by subject

        batched_indices = []
        buffer = []
        for index in indices:
            if (len(buffer) == self.batch_size) or (
                len(buffer) > 0 and buffer[0][0] != index[0]
            ):
                batched_indices.append(buffer)
                buffer = []
            buffer.append(index)
        if len(buffer) > 0 and not self.drop_last:
            batched_indices.append(buffer)

        if self.shuffle:
            np.random.shuffle(batched_indices)
        yield from batched_indices

    def __len__(self):
        if self._length >= 0:
            return self._length
        total = 0
        indices = self.data_source._get_keys(by=self.by)
        indices.sort(key=lambda x: x[0])  # sort by subject

        buffer = []
        for index in indices:
            if (len(buffer) == self.batch_size) or (
                len(buffer) > 0 and buffer[0][0] != index[0]
            ):
                total += 1
                buffer = []
            buffer.append(index)
        if len(buffer) > 0 and not self.drop_last:
            total += 1
        self._length = total
        return total


class MultiSourceDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        base_sampler: MultiSourceBatchSampler,
        num_replicas=None,
        rank=None,
        seed=0,
    ):
        self.base_sampler = base_sampler
        shuffle = self.base_sampler.shuffle
        drop_last = self.base_sampler.drop_last
        super().__init__(
            base_sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self):
        base_indices = list(self.base_sampler.__iter__())
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        indices = [base_indices[i] for i in indices]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class fMRIInstructionDataset(Dataset):
    def __init__(
        self,
        dataset,
        instruction_dataset,
        tokenizer,
        generate_prompt=False,
        source=None,
        whole_brain=False,
        split="none",
        mixup=True,
        train_samples=-1,
        image_processor=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.source = source
        self.split = split
        assert split in ["train", "val", "test"], split

        self.instruction_dataset = instruction_dataset

        self.tokenizer = tokenizer
        self.generate_prompt = generate_prompt
        self.whole_brain = whole_brain
        self.mixup = mixup
        self.train_samples = train_samples
        self.image_processor = image_processor

    def process_sample(self, voxels, sample_or_coco_id, chat):
        output = {
            "chat": (
                self.format_chat(chat["chat"])
                if self.tokenizer is not None
                else chat["chat"]
            ),
            "voxels": torch.from_numpy(voxels),
            "sample_or_coco_id": sample_or_coco_id,
            "question_id": chat.get("question_id", None),
            "answers": chat.get("answers", None),
        }
        if self.image_processor is not None:
            image = load_coco_image(sample_or_coco_id, "data/coco")
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor
                    #     transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop((256, 256)),  # Crop the image to 224x224
                ]
            )
            image = transform(image)
            image = self.image_processor(image)
            output["image"] = image

        if self.generate_prompt:
            if self.tokenizer is None:
                output["prompt"] = chat["chat"][:-1]
            else:
                output["prompt"] = self.format_chat(chat["chat"], is_generation=True)
                output["prompt"]["labels"] = output["chat"]["input_ids"][
                    len(output["prompt"]["input_ids"]) :
                ]

        output["source"] = chat["from"]
        output["original_chat"] = chat

        return output

    def format_chat(self, chat, is_generation=False):
        if is_generation:
            chat = chat[:-1]
            assert chat[-1]["role"] == "user"

        return_assistant_tokens_mask = True
        output = apply_chat_template(
            chat=chat,
            tokenizer=self.tokenizer,
            add_generation_prompt=is_generation,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
        )
        assert isinstance(output["input_ids"][0], int)
        if not is_generation:
            if return_assistant_tokens_mask:
                output["labels"] = [
                    IGNORE_INDEX if mask == 0 else output["input_ids"][i]
                    for i, mask in enumerate(output["assistant_masks"])
                ]
            else:
                output["labels"] = output["input_ids"]
        return output

    def _get_keys(self, by="sample"):
        indices = []
        for subj, dataset_subj in self.dataset.items():
            dataset_subj_items = sorted(list(dataset_subj.items()))
            for sample_key, sample in dataset_subj_items:
                indices.append((subj, sample_key, int(sample["coco_id"][()])))

        if by == "coco":
            indices_group_by_coco_id = defaultdict(list)
            for subj, sample_key, coco_id in indices:
                indices_group_by_coco_id[(subj, coco_id)].append(sample_key)
            indices_group_by_coco_id = [
                (subj, tuple(sample_keys), instruction_key)
                for (
                    subj,
                    instruction_key,
                ), sample_keys in indices_group_by_coco_id.items()
            ]
            new_indices = indices_group_by_coco_id
        elif by == "sample":
            indices_group_by_sample_key = [
                (subj, sample_key, f"{subj}-{self.split}-{sample_key}")
                for subj, sample_key, _ in indices
            ]
            new_indices = indices_group_by_sample_key

        indices = []  # expand with chat idx
        for subj, sample_key_or_keys, instruction_key in new_indices:
            if self.split == "train":
                if (
                    instruction_key in self.instruction_dataset
                    and len(self.instruction_dataset[instruction_key]) > 0
                ):
                    indices.append((subj, sample_key_or_keys, instruction_key, -1))
            else:
                if instruction_key in self.instruction_dataset:
                    for i, instruction in enumerate(
                        self.instruction_dataset[instruction_key]
                    ):
                        indices.append((subj, sample_key_or_keys, instruction_key, i))

        if self.split != "train":
            indices.sort(key=lambda x: str(x[1]))
        else:
            if self.train_samples > 0:
                indices.sort()
                indices = indices[: self.train_samples]
                indices = [
                    (
                        index[0],
                        index[1][0] if by == "coco" else index[1],
                        index[2],
                        index[3],
                    )
                    for index in indices
                ]
                random.shuffle(indices)

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subj, sample_key_or_keys, instruction_key, chat_idx = idx
        dataset_subj = self.dataset[subj]
        if self.whole_brain:
            grp_name = dataset_subj.name.split("/")[-1]

        def adapt_voxels(voxels):
            if self.split == "train":
                if self.mixup:
                    alpha = 0.4
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(voxels.shape[0])
                    voxels = lam * voxels + (1 - lam) * voxels[index]
                repeat_index = random.randint(0, voxels.shape[0] - 1)
                voxels = voxels[repeat_index]
            else:
                voxels = voxels.mean(0)
            return voxels

        if isinstance(sample_key_or_keys, tuple):
            voxels = np.stack(
                [dataset_subj[k]["visual_voxels"][()] for k in sample_key_or_keys],
                axis=0,
            )
            voxels = adapt_voxels(voxels)
            if self.whole_brain:
                whole_brain = np.stack(
                    [
                        np.load(f"data/whole_brain/{subj}/{grp_name}/{k}.npy")
                        for k in sample_key_or_keys
                    ],
                    axis=0,
                )
                whole_brain = adapt_voxels(whole_brain)
        else:
            data = dataset_subj[sample_key_or_keys]
            voxels = data["visual_voxels"][()]
            if self.whole_brain:
                whole_brain = np.load(
                    f"data/whole_brain/{subj}/{grp_name}/{sample_key_or_keys}.npy"
                )

        chats = self.instruction_dataset[instruction_key]
        if chat_idx < 0:
            source = random.choice(list(chats.keys()))
            chat = random.choice(chats[source])
        else:
            chat = chats[chat_idx]
        output = self.process_sample(voxels, instruction_key, chat)
        if self.whole_brain:
            output["voxels"] = torch.from_numpy(whole_brain)
        output["subject"] = subj
        return output

    def group(self):
        indices = []

        coco_to_sample_id = defaultdict(lambda: defaultdict(set))
        for key in self.indices:
            subj, sample_id, _ = key
            coco_to_sample_id[subj][
                int(self.dataset[subj][sample_id]["coco_id"][()])
            ].add(sample_id)

        for subj, coco_to_sample_id_subj in coco_to_sample_id.items():
            for coco_id, sample_ids in coco_to_sample_id_subj.items():
                assert len(sample_ids) == 3
                coco_to_sample_id[subj][coco_id] = tuple(sample_ids)

        for subj, coco_to_sample_id_subj in coco_to_sample_id.items():
            for coco_id, sample_ids in coco_to_sample_id_subj.items():
                for i, instruction in enumerate(self.instruction_dataset[coco_id]):
                    indices.append((subj, sample_ids, i))
        self.indices = indices
        return self


class ValueWrapper:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        if key == ():
            return self.value
        raise NotImplementedError


class MindEyeSample:
    def __init__(self, path, ind):
        self.path = path
        self.ind = ind

    def __getitem__(self, key):
        if key == "visual_voxels":
            return np.load(self.path + ".nsdgeneral.npy")[self.ind]
        elif key == "coco_id":
            coco_id = np.load(self.path + ".coco73k.npy").item()
            coco_id = nsd_to_image_id[coco_id]
            return ValueWrapper(coco_id)
        else:
            raise


class MindEyeSamples:
    def __init__(self, subj=1, path="data/webdataset_avg_split", split="train"):
        assert split in ["train", "val", "test"]
        self.path = path
        self.split = split
        files = glob(os.path.join(path, split, f"subj0{subj}", "*.nsdgeneral.npy"))
        assert len(files) > 0
        sample_keys = []
        for file in files:
            sample_key = file.replace(".nsdgeneral.npy", "")
            for i in range(3):
                sample_keys.append(sample_key + f"_{i}")
        self.sample_keys = sample_keys

    def __getitem__(self, key):
        sample_key, ind = key.rsplit("_", 1)
        ind = int(ind)
        return MindEyeSample(sample_key, ind)

    def __len__(self):
        return len(self.sample_keys)

    def items(self):
        for key in self.sample_keys:
            yield key, self[key]


def get_mindeye_dataset(subj_list):
    dataset = {}
    for split in ["train", "val", "test"]:
        dataset[split] = {}
        for subj in subj_list:
            dataset[split][f"subject_{subj}"] = MindEyeSamples(subj, split=split)
    return dataset


def get_mindeye_dataloader(subj=1, split="val"):
    import webdataset as wds
    info = pd.read_csv(
        os.path.join("~/data/fMRI-reconstruction-NSD", "nsd_stim_info_merged.csv")
    )
    nsd_to_image_id = (
        info[["Unnamed: 0", "cocoId"]]
        .set_index("Unnamed: 0", drop=True)
        .to_dict()["cocoId"]
    )

    def to_dict(sample):
        return {
            "visual_voxels": sample["nsdgeneral.npy"],
            "coco_id": nsd_to_image_id[sample["coco73k.npy"].item()],
        }

    val_url = f"data/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    if split == "val":
        data = (
            wds.WebDataset(val_url, resampled=False, nodesplitter=lambda x: x)
            .decode("torch")
            .map(to_dict)
        )
    else:
        raise NotImplementedError

    return DataLoader(data, batch_size=1, num_workers=0, shuffle=False)


class ConcatDataLoader:
    def __init__(self, loaders, max_batch_for_each=1e8):
        self.loaders = loaders
        self.max_batch_for_each = max_batch_for_each

    def __iter__(self):
        for loader in self.loaders:
            for i, batch in enumerate(loader):
                if i > self.max_batch_for_each:
                    continue
                yield batch

    def __len__(self):
        length = 0
        for loader in self.loaders:
            length += min(self.max_batch_for_each, len(loader))
        return length


def train_val_split_by_coco_id(data, num_val, subject_i, seed):
    cache_path = (
        "data/split/default.json" if seed < 0 else f"data/split/seed{seed}.json"
    )
    cache_path = cache_path.replace(".json", f"-{subject_i}.json")
    if num_val < 1:
        raise NotImplementedError

    coco_to_sample_id = defaultdict(set)
    for key in data:
        coco_id = data[key]["coco_id"][()].item()
        coco_to_sample_id[coco_id].add(key)

    from filelock import FileLock

    lock = FileLock(cache_path + ".lock")
    with lock:
        if os.path.exists(cache_path) and False:
            split = json.load(open(cache_path))
            train_keys = split["train"]
            val_keys = split["val"]
            json.dump({"train": train_keys, "val": val_keys}, open(cache_path, "w"))
        else:
            keys = list(coco_to_sample_id.keys())
            random.Random(seed).shuffle(keys)
            val_keys = keys[:num_val]
            train_keys = keys[num_val:]
   
    assert len(coco_to_sample_id) == len(train_keys) + len(val_keys)
    train = {}
    val = {}
    for coco_id in val_keys:
        sample_ids = coco_to_sample_id[coco_id]
        val.update({k: data[k] for k in sample_ids})
    for coco_id in train_keys:
        sample_ids = coco_to_sample_id[coco_id]
        train.update({k: data[k] for k in sample_ids})
    return train, val


class fMRIInstructionDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        nsd_version: int = 0,
        task: str = None,
        exclude: str = None,
        subjects: list = [1],
        whole_brain: bool = False,
        group_by_coco: bool = True,
        mixup: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        train_samples: int = -1,
        split_val: bool = False,
        split_seed: int = -1,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        if nsd_version == 0:
            self._files = {
                f"subject_{subj}": h5py.File(f"data/nsd0/subject_{subj}.h5", "r")
                for subj in subjects
            }
            self.file = {
                "train": {
                    f"subject_{subj}": self._files[f"subject_{subj}"]["train"]
                    for subj in subjects
                },
                "test": {
                    f"subject_{subj}": self._files[f"subject_{subj}"]["test"]
                    for subj in subjects
                },
            }
            self.file = {
                "train": {
                    f"subject_{subj}": self._files[f"subject_{subj}"]["train"]
                    for subj in subjects
                },
                "val": {},
                "test": {
                    f"subject_{subj}": self._files[f"subject_{subj}"]["test"]
                    for subj in subjects
                },
            }
            if split_val:
                for subject_i in self.file["train"]:
                    train, val = train_val_split_by_coco_id(
                        self.file["train"][subject_i], 900, subject_i, seed=split_seed
                    )
                    self.file["train"][subject_i] = train
                    self.file["val"][subject_i] = val
        elif nsd_version == 1:
            assert not whole_brain
            self.file = get_mindeye_dataset(subjects)

        self.nsd_version = nsd_version
        self.whole_brain = whole_brain
        self.task = task
        self.exclude = exclude
        self.subjects = subjects
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.split_val = split_val

        self.collate_fn = DataCollator(tokenizer)
        self.group_by = "coco" if group_by_coco else "sample"
        self.mixup = mixup

    def setup(self, stage):
        self.instruction_dataset = (
            InstructionDatasetBuilder(
                self.file, subjects=self.subjects, whole_brain=self.whole_brain
            )
            .filter(self.task)
            .exclude(self.exclude)
            .load()
        )

        instruction_dataset = self.instruction_dataset.build()

        if stage == "fit":
            self.train_data = fMRIInstructionDataset(
                self.file["train"],
                instruction_dataset=instruction_dataset,
                tokenizer=self.tokenizer,
                train_samples=self.train_samples,
                split="train",
                whole_brain=self.whole_brain,
                mixup=self.mixup,
            )

        if stage == "validate" or stage == "fit":
            self.val_data = [
                self.get_data_by_source(
                    self.file["test"], source, split="val", generate_prompt=True
                )
                for source in self.instruction_dataset.datasets.keys()
            ]

        if stage == "test" or stage == "predict":
            self.test_data = [
                self.get_data_by_source(
                    self.file["test"], source, split="test", generate_prompt=True
                )
                for source in self.instruction_dataset.datasets.keys()
                if source not in ["llava", "visdial", "lvis"]
            ]

    def get_data_by_source(self, data, source, **kwargs):
        instruction_dataset = self.instruction_dataset.filter(source).unique().build()
        if len(instruction_dataset) == 0:
            raise ValueError(f"{source} has no instructions in the dataset")

        dataset = fMRIInstructionDataset(
            data,
            instruction_dataset=instruction_dataset,
            tokenizer=self.tokenizer,
            source=source,
            whole_brain=self.whole_brain,
            mixup=self.mixup,
            **kwargs,
        )

        # if source in IMAGE_SOURCES:
        #     dataset = dataset.group()
        return dataset

    def may_wrap_distributed(self, sampler):
        if dist.is_initialized():
            sampler = MultiSourceDistributedSampler(sampler)
        return sampler

    def train_dataloader(self):
        batch_sampler = MultiSourceBatchSampler(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            by=self.group_by,
        )
        train_loader = DataLoader(
            self.train_data,
            batch_sampler=self.may_wrap_distributed(batch_sampler),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        if isinstance(self.val_data, list):
            loaders = []
            for val_data in self.val_data:
                batch_sampler = MultiSourceBatchSampler(
                    val_data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    by=self.group_by,
                )
                loaders.append(
                    DataLoader(
                        val_data,
                        batch_sampler=self.may_wrap_distributed(batch_sampler),
                        num_workers=self.num_workers,
                        collate_fn=self.collate_fn,
                    )
                )
            return ConcatDataLoader(
                loaders, max_batch_for_each=200 if self.task is None else int(1e8)
            )
        else:
            batch_sampler = MultiSourceBatchSampler(
                self.val_data, drop_last=False, by=self.group_by
            )
            return DataLoader(
                self.val_data,
                batch_sampler=self.may_wrap_distributed(batch_sampler),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self):
        if isinstance(self.test_data, list):
            loaders = []
            for test_data in self.test_data:
                batch_sampler = MultiSourceBatchSampler(
                    test_data,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    by=self.group_by,
                )
                loaders.append(
                    DataLoader(
                        test_data,
                        batch_sampler=self.may_wrap_distributed(batch_sampler),
                        num_workers=self.num_workers,
                        collate_fn=self.collate_fn,
                    )
                )
            return ConcatDataLoader(loaders, max_batch_for_each=2000000000000)
        else:
            batch_sampler = MultiSourceBatchSampler(
                self.test_data, drop_last=False, by=self.group_by
            )
            return DataLoader(
                self.test_data,
                batch_sampler=self.may_wrap_distributed(batch_sampler),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

    def predict_dataloader(self):
        if isinstance(self.test_data, list):
            loaders = []
            for test_data in self.test_data:
                batch_sampler = MultiSourceBatchSampler(
                    test_data,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    by=self.group_by,
                )
                loaders.append(
                    DataLoader(
                        test_data,
                        batch_sampler=self.may_wrap_distributed(batch_sampler),
                        num_workers=self.num_workers,
                        collate_fn=self.collate_fn,
                    )
                )
            return ConcatDataLoader(loaders)
        else:
            batch_sampler = MultiSourceBatchSampler(
                self.test_data, drop_last=False, by=self.group_by
            )
            return DataLoader(
                self.test_data,
                batch_sampler=self.may_wrap_distributed(batch_sampler),
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )