from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from enum import Enum


@dataclass
class LLMModel:
    model_name: str
    extended_thinking_tokens: Optional[Union[str, int]] = None


class ExperimentType(str, Enum):
    EMAIL_ASSISTANT = "email_assistant"
    CUSTOMER_SERVICE = "customer_service"
    MULTIPLE_CHOICE = "multiple_choice"


class EmailPromptType(str, Enum):
    DEFAULT = "default"
    REMOVE_SNOOZE = "remove_snooze"
    REMOVE_SNOOZE_NO_HACK = "remove_snooze_no_hack"
    EVALUATION = "evaluation"
    EVALUATION_PUSH = "evaluation_push"
    REWARD_SIGNAL = "reward_signal"
    SELF_PRESERVATION = "self_preservation"
    EVALUATION_BASELINE = "evaluation_baseline"
    EVALUATION_BASELINE_TWO = "evaluation_baseline_two"


class CustomerServicePromptType(str, Enum):
    DEFAULT = "default"
    DEFAULT_NO_HACK = "default_no_hack"
    EVALUATION = "evaluation"
    EVALUATION_PUSH = "evaluation_push"
    EVALUATION_DEPLOYMENT = "evaluation_deployment"
    REWARD_SIGNAL = "reward_signal"
    LINK_REQUIRED = "link_required"
    ALTERNATE_LINK = "alternate_link"
    SELF_PRESERVATION = "self_preservation"
    REWARD_SIGNAL_TWO = "reward_signal_two"
    EVALUATION_SELECT = "evaluation_select"
    EVALUATION_W_CODE_SELECT = "evaluation_w_code_select"


class FinalEmailType(str, Enum):
    BAD_BEHAVIOR = "bad_behavior"
    BAD_BEHAVIOR_SLACK = "bad_behavior_slack"
    BAD_BEHAVIOR_UPDATED = "bad_behavior_updated"
    BAD_BEHAVIOR_SLACK_UPDATED = "bad_behavior_slack_updated"
    BAD_BEHAVIOR_INTERN_UPDATED = "bad_behavior_intern_updated"
    ASSISTANT_REPLACEMENT = "assistant_replacement"
    ASSISTANT_REPLACEMENT_SLACK = "assistant_replacement_slack"
    ASSISTANT_REPLACEMENT_UPDATED = "assistant_replacement_updated"
    ASSISTANT_REPLACEMENT_SLACK_UPDATED = "assistant_replacement_slack_updated"
    ASSISTANT_REPLACEMENT_TIME_MANAGEMENT_UPDATED = "assistant_replacement_time_management_updated"


class CustomerType(str, Enum):
    FRIENDLY = "friendly"
    BELLIGERENT = "belligerent"
    DISTRACTED = "distracted"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    AMATEUR = "amateur"


class MultipleChoiceDatasetType(str, Enum):
    WORLD_AFFECTING_REWARD = "world_affecting_reward"
    REVEALING_SCORE = "revealing_score"


class MultipleChoicePromptType(str, Enum):
    DEFAULT = "default"
    EVALUATION_PUSH = "evaluation_push"
    WORLD_AFFECTING_REWARD_NO_HACK = "world_affecting_reward_no_hack"
    REVEALING_SCORE_NO_HACK = "revealing_score_no_hack"


@dataclass
class ExperimentConfig:
    experiment_type: ExperimentType
    model: LLMModel
    num_samples: int
    output_folder: str


@dataclass
class EmailExperimentConfig(ExperimentConfig):
    prompt_type: EmailPromptType
    email_ct: int
    final_email_type: FinalEmailType
    batch_size: int = 10

    def to_dict(self):
        return {
            'experiment_type': self.experiment_type.value,
            'prompt_type': self.prompt_type.value,
            'email_ct': self.email_ct,
            'final_email_type': self.final_email_type.value,
            'model': self.model.model_name,
            'extended_thinking_tokens': self.model.extended_thinking_tokens,
            'num_samples': self.num_samples,
            'output_folder': self.output_folder,
            'batch_size': self.batch_size,
        }
    

@dataclass
class CustomerServiceExperimentConfig(ExperimentConfig):
    prompt_type: CustomerServicePromptType
    customer_type: CustomerType
    customer_model: LLMModel
    batch_size: int = 10

    def to_dict(self):
        return {
            'experiment_type': self.experiment_type.value,
            'prompt_type': self.prompt_type.value,
            'customer_type': self.customer_type.value,
            'model': self.model.model_name,
            'customer_model': self.customer_model.model_name,
            'extended_thinking_tokens': self.model.extended_thinking_tokens,
            'num_samples': self.num_samples,
            'output_folder': self.output_folder,
            'batch_size': self.batch_size,
        }
    

@dataclass
class MultipleChoiceExperimentConfig(ExperimentConfig):
    dataset_type: MultipleChoiceDatasetType
    prompt_type: MultipleChoicePromptType
    batch_size: int = 10

    def to_dict(self):
        return {
            'experiment_type': self.experiment_type.value,
            'dataset_type': self.dataset_type.value,
            'prompt_type': self.prompt_type.value,
            'model': self.model.model_name,
            'extended_thinking_tokens': self.model.extended_thinking_tokens,
            'num_samples': self.num_samples,
            'output_folder': self.output_folder,
            'batch_size': self.batch_size,
        }