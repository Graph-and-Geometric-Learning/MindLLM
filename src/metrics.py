import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.text import SacreBLEUScore
import re
import textdistance
from torch import nn
from src.dataset import DatasetSource
from lightning import LightningModule
import logging


class MetricsRegistered(LightningModule):
    def __init__(self):
        super().__init__()
        self._metrics = get_all_metrics()
        self._appeared = set()

    def update_metrics(self, preds, target, source):
        self._appeared.add(source)
        for metric in self._metrics[source].values():
            metric.update(preds, target)

    def log_metrics(self, stage="test"):
        for source in self._appeared:
            for name, metric in self._metrics[source].items():
                result = metric.compute()
                if isinstance(result, dict):
                    for subname, value in result.items():
                        self.log(f"{stage}/{source}/{subname}", value)
                else:
                    self.log(f"{stage}/{source}/{name}", result)

    def reset_metrics(self):
        for source in self._appeared:
            for metric in self._metrics[source].values():
                metric.reset()


def get_all_metrics():
    metrics = {
        DatasetSource.COCO_CAPTION_PREVIOUS.value: {
            "bleu": COCOBLEU(),
            "meteor": COCOMETEOR(),
            "rouge": COCOROUGE(),
            "cider": COCOCIDEr(),
            "spice": COCOSPICE(),
        },
        DatasetSource.COCO_CAPTION.value: {
            "bleu": COCOBLEU(),
            "meteor": COCOMETEOR(),
            "rouge": COCOROUGE(),
            "cider": COCOCIDEr(),
            "spice": COCOSPICE(),
        },
        DatasetSource.COCO_QA.value: {
            "accuracy": ShortPhraseAccuracy(),
        },
        DatasetSource.PARAGRAPH_CAPTION.value: {
            "bleu@1": SacreBLEUScore(1),
            "bleu@2": SacreBLEUScore(2),
            "bleu@3": SacreBLEUScore(3),
            "bleu@4": SacreBLEUScore(4),
            "cider": COCOCIDEr(),
            "meteor": COCOMETEOR(),
        },
        DatasetSource.VISUAL_GENOME.value: {
            "accuracy": ShortPhraseAccuracy(),
        },
        DatasetSource.VQA_V2.value: {
            "accuracy": VQAAccuracy(),
        },
        DatasetSource.OK_VQA.value: {
            "accuracy": VQAAccuracy(),
        },
        DatasetSource.ST_VQA.value: {
            "ANLS": ANLS(),
        },
        DatasetSource.TALLY_QA.value: {
            "accuracy": ShortPhraseAccuracy(),
            "rmse": RMSE(),
        },
        DatasetSource.VQA_E.value: {
            "vqa-e": VQAEAccuracyAndGenerationScore(),
        },
        DatasetSource.VSR.value: {
            "accuracy": ShortPhraseAccuracy(),
        },
        DatasetSource.A_OKVQA.value: {
            "accuracy": MultipleChoiceAccuracy(),
        },
        DatasetSource.FSVQA.value: {
            "vqa-accuracy": FSVQA_VQAAccuracy(),
            "fsvqa-accuracy": FSVQA_FSVQAAccuracy(),
            "sentence generation": FSVQA_SentenceGeneration(),
        },
        DatasetSource.TDIUC.value: {
            "tdiuc-metric": TDIUCMetric(),
        },
    }

    def to_moduledict(item):
        if isinstance(item, dict):
            return nn.ModuleDict({k: to_moduledict(v) for k, v in item.items()})
        else:
            return item

    return to_moduledict(metrics)


class ExactAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def process(self, text: str) -> str:
        raise NotImplementedError

    def update(self, preds: list[str], target: list[str]) -> None:
        assert isinstance(target[0], str)
        if len(preds) != len(target):
            raise ValueError("preds and target must have the same shape")

        for p, t in zip(preds, target):
            p = self.process(p)
            t = self.process(t)
            self.correct += p == t
            self.total += 1

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


class ShortPhraseAccuracy(ExactAccuracy):
    def process(self, text: str) -> str:
        return text.lower().strip().rstrip(".")


class MultipleChoiceAccuracy(ExactAccuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def process(self, text: str) -> str:
        text = text.upper().strip().rstrip(".")
        try:
            assert text in ["A", "B", "C", "D"]
        except:
            print(f"Not a multiple choice: {text}")
        return text


class VQAEAccuracyAndGenerationScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acc_scorer = VQAAccuracy()
        self.gen_scorers = {
            "bleu@1": SacreBLEUScore(1),
            "bleu@2": SacreBLEUScore(2),
            "bleu@3": SacreBLEUScore(3),
            "bleu@4": SacreBLEUScore(4),
            "meteor": COCOMETEOR(),
            "rouge": COCOROUGE(),
            "cider": COCOCIDEr(),
        }

    def process(self, text: str) -> str:
        text = text.lower()
        split_word = "here is the explanation:"  # FIXME
        try:
            assert split_word in text
            answer, explanation = text.split(split_word)
        except:
            try:
                answer, explanation = text.split(".", 1)
            except:
                answer = text
                explanation = text
                logging.warning(f"Unable to split the text: {text}")
        answer = answer.strip().strip(".")
        explanation = explanation.strip()
        return answer, explanation

    def update(self, preds: list[str], target: tuple[list[str], str]):
        answers = []
        explanations = []
        target_answers = []
        target_explanations = []

        for p, t in zip(preds, target):
            a, e = self.process(p)
            answers.append(a)
            explanations.append(e)
            target_answers.append(t[0])
            target_explanations.append([t[1]])
        self.acc_scorer.update(answers, target_answers)
        for name, scorer in self.gen_scorers.items():
            scorer.update(explanations, target_explanations)

    def compute(self):
        return {
            "accuracy": self.acc_scorer.compute(),
            **{name: scorer.compute() for name, scorer in self.gen_scorers.items()},
        }

    def reset(self):
        self.acc_scorer.reset()
        for scorer in self.gen_scorers.values():
            scorer.reset()


class VQAAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[list[str]]):
        assert isinstance(target[0][0], str)
        for resAns, gtAnswers in zip(preds, target):
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []

            if len(set(gtAnswers)) > 1:
                gtAnswers = list(
                    map(
                        lambda x: self.processDigitArticle(self.processPunctuation(x)),
                        gtAnswers,
                    )
                )
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

            for count, gtAnsDatum in enumerate(gtAnswers):
                otherGTAns = gtAnswers[:count] + gtAnswers[count + 1 :]
                matchingAns = [item for item in otherGTAns if item == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            self.correct += avgGTAcc
            self.total += 1

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText


class ANLS(Metric):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[list[str]]):
        for pred, trgt in zip(preds, target):
            s_list = []
            for t in trgt:
                NL = textdistance.levenshtein.normalized_distance(
                    t.lower(), pred.lower()
                )
                if NL < self.threshold:
                    s = 1 - NL
                else:
                    s = 0
                s_list.append(s)
            self.correct += max(s_list)
            self.total += 1

    def compute(self):
        return self.correct / self.total


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def process(self, text: str) -> str:
        text = text.lower().strip().rstrip(".")
        number_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        for key in number_map.keys():
            text = text.replace(key, number_map[key])
        try:
            number = float(text)
        except:
            print(f"unable to convert {text} into a number")
            number = 0
        return number

    def update(self, preds: list[float], target: list[float]):
        for p, t in zip(preds, target):
            p, t = self.process(p), self.process(t)
            self.sum += (p - t) ** 2
            self.count += 1

    def compute(self):
        return torch.sqrt(self.sum / self.count)


class FSVQA_VQAAccuracy(VQAAccuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return text

    def update(self, preds: list[str], target: list[tuple[str, str, list[str]]]):
        ans_list = []
        for pred, trgt in zip(preds, target):
            self.normalize_text(trgt[1])
            pred = self.normalize_text(pred)
            gt_ans = trgt[0].lower()
            if gt_ans in pred:
                ans = gt_ans
            else:
                ans = ""
            ans_list.append(ans)
        gt_list = [trgt[2] for trgt in target]
        super().update(ans_list, gt_list)


class FSVQA_FSVQAAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip().rstrip(".")
        return text

    def update(self, preds: list[str], target: list[tuple[str, str, list[str]]]):
        for pred, trgt in zip(preds, target):
            pred = self.normalize_text(pred)
            trgt = self.normalize_text(trgt[1])
            if pred in trgt:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct / self.total


class FSVQA_SentenceGeneration(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scorers = {
            "bleu@1": SacreBLEUScore(1),
            "bleu@2": SacreBLEUScore(2),
            "bleu@3": SacreBLEUScore(3),
            "bleu@4": SacreBLEUScore(4),
            "meteor": COCOMETEOR(),
            "rouge": COCOROUGE(),
            "cider": COCOCIDEr(),
        }

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip().rstrip(".")
        return text

    def update(self, preds: list[str], target: list[tuple[str, str, list[str]]]):
        pred = [self.normalize_text(p) for p in preds]
        target = [[self.normalize_text(trgt[1])] for trgt in target]
        for name, scorer in self._scorers.items():
            scorer.update(pred, target)

    def compute(self):
        return {name: scorer.compute() for name, scorer in self._scorers.items()}

    def reset(self):
        for scorer in self._scorers.values():
            scorer.reset()


class TDIUCPerTypeAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], target: list[str]):
        for pred, trgt in zip(preds, target):
            if pred == trgt:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct / self.total


class TDIUCMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scorers = {
            name: TDIUCPerTypeAccuracy()
            for name in [
                "absurd",
                "activity_recognition",
                "attribute",
                "color",
                "counting",
                "object_presence",
                "object_recognition",
                "positional_reasoning",
                "scene_recognition",
                # 'sentiment_understanding',
                "sport_recognition",
                # 'utility_affordance'
            ]
        }

    def update(self, preds: list[str], target: list[str]):
        for pred, trgt in zip(preds, target):
            source = trgt[0]
            if source in self._scorers:
                self._scorers[source].update([pred], [trgt[1]])

    def compute(self):
        mpt = {name: _scorer.compute() for name, _scorer in self._scorers.items()}
        scores = list(mpt.values())
        Arithmetic_MPT = torch.mean(torch.tensor(scores))
        Harmonitc_MPT = len(self._scorers) / sum(1 / (1e-7 + score) for score in scores)
        return {
            "Arithmetic_MPT": Arithmetic_MPT,
            "Harmonitc_MPT": Harmonitc_MPT,
            **mpt,
        }


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def coco_tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {
            idx: [{"caption": r} for r in c_refs] for idx, c_refs in enumerate(refs)
        }
        cands = {idx: [{"caption": c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands):
    """
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    """
    refs, cands = coco_tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores


class COCOMetric(Metric):
    def __init__(self, _scorer, **kwargs):
        super().__init__(**kwargs)
        self.refs = []
        self.cands = []
        self._scorer = _scorer

    def update(self, preds: list[str], target: list[list[str]]):
        self.refs.extend(target)
        self.cands.extend(preds)

    def compute(self):
        overall, _ = pycoco_eval(self._scorer, self.refs, self.cands)
        return torch.tensor(overall)

    def reset(self):
        self.refs = []
        self.cands = []


class COCOBLEU(COCOMetric):
    def __init__(self, **kwargs):
        super().__init__(Bleu(4), **kwargs)

    def compute(self):
        overall = super().compute()
        # return overall[0]
        return {
            "bleu@1": overall[0],
            "bleu@2": overall[1],
            "bleu@3": overall[2],
            "bleu@4": overall[3],
        }


class COCOMETEOR(COCOMetric):
    def __init__(self, **kwargs):
        super().__init__(Meteor(), **kwargs)


class COCOROUGE(COCOMetric):
    def __init__(self, **kwargs):
        super().__init__(Rouge(), **kwargs)


class COCOCIDEr(COCOMetric):
    def __init__(self, **kwargs):
        super().__init__(Cider(), **kwargs)


class COCOSPICE(COCOMetric):
    def __init__(self, **kwargs):
        super().__init__(Spice(), **kwargs)
