from typing import Literal, List
from pydantic import BaseModel, Field, model_validator, field_validator


class ExpConfig(BaseModel):
    model_config = {"extra": "allow"}


    d_is_training: int = Field(
        1,
        description="Execution mode: 1 = train + test, 0 = test only (no training)."
    )

    # -------------------------
    # Dataset Information.
    # -------------------------

    d_data: str = Field(
        ...,
        description="Dataset identifier (used for logging and experiment naming)."
    )

    d_root_path: str = Field(
        ...,
        description="Root directory containing the dataset files."
    )

    d_data_path: str = Field(
        ...,
        description="Name of the dataset file (e.g., ETTh1.csv) located in root_path."
    )

    d_forecast_type: Literal["M", "S", "MS"] = Field(
        "S",
        description=(
            "Forecasting setup:\n"
            "M  = multivariate → multivariate\n"
            "S  = univariate → univariate\n"
            "MS = multivariate → univariate"
        )
    )

    d_freq: Literal["h", "min", "s"] = Field(
        "h",
        description=(
            "Data is either: per hour (h), per minute (min), per second (s)"
        )
    )

    d_target: int = Field(
        -1,
        description=(
            "Index of target feature (used only for MS or S forecasting). "
            "Ignored when using full multivariate (M)."
        )
    )

    d_embed: Literal["timeF", "fixed", "learned"] = Field(
        "timeF",
        description="Time feature encoding method."
    )

    d_checkpoint_path: str = Field(
        "./checkpoints/",
        description="Directory where model checkpoints will be saved."
    )

    d_train_stride: int = Field(
        5,
        description="Stride used when generating sliding windows from training time series data."
    )

    d_test_stride: int = Field(
        1,
        description="Stride used when generating sliding windows from validating and testing time series data."
    )

    d_inverse_transform: bool = Field(
        False,
        description="Whether to apply inverse scaling to predictions before evaluation."
    )

    # -------------------------
    # Sequence lengths
    # -------------------------
    d_seq_len: int = Field(
        96,
        gt=0,
        description="Length of the historical input sequence."
    )

    d_label_len: int = Field(
        48,
        ge=0,
        description="Length of the known context fed to decoder (used in encoder-decoder models)."
    )

    d_pred_len: int = Field(
        96,
        gt=0,
        description="Prediction horizon (number of future time steps to forecast)."
    )

    # -------------------------
    # Feature dimensions
    # -------------------------
    d_in_features: int = Field(
        ...,
        gt=0,
        description="Number of input variables (channels) in the time series."
    )

    d_out_features: int = Field(
        ...,
        gt=0,
        description="Number of output variables to predict."
    )

    # -------------------------
    # Output behavior
    # -------------------------
    d_output_attention: bool = Field(
        False,
        description="Whether the model returns attention weights along with predictions."
    )


    # -------------------------
    # Model identity
    # -------------------------
    d_model: str = Field(
        ...,
        description="Model name (e.g., DLinear, iTransformer, PatchTST)."
    )

    d_model_input_type: str = Field(
        ...,
        description="Input format expected by the model (e.g., x_only, x_mark_dec_incld)."
    )


    d_num_workers: int = Field(
        10,
        ge=0,
        description="Number of worker processes for data loading (0 = main process only)."
    )

    d_train_epochs: int = Field(
        100,
        gt=0,
        description="Total number of training epochs."
    )

    d_batch_size: int = Field(
        32,
        gt=0,
        description="Batch size used during training."
    )

    d_patience: int = Field(
        20,
        ge=0,
        description="Early stopping patience (number of epochs with no improvement before stopping)."
    )

    d_learning_rate: float = Field(
        1e-3,
        gt=0,
        description="Initial learning rate for the optimizer."
    )

    # d_des: str = Field(
    #     "test",
    #     description="Experiment description or tag used for logging and checkpoint naming."
    # )

    d_loss: Literal["mse", "mae"] = Field(
        "mse",
        description="Loss function used for training."
    )

    d_auxillary_loss: bool = Field(
        False,
        description="Whether to include auxiliary loss from model (if available) during training."
    )

    d_lradj: str = Field(
        "type1",
        description="Learning rate adjustment strategy (scheduler type)."
    )

    d_pct_start: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of training steps used for warm-up (if applicable)."
    )


    d_use_gpu: bool = Field(
        False,
        description="Enable GPU acceleration if CUDA is available."
    )

    d_gpu: int = Field(
        0,
        ge=0,
        description="Index of the primary GPU (e.g., 0 for cuda:0). Must be included in d_devices."
    )

    d_use_multi_gpu: bool = Field(
        False,
        description="Enable multi-GPU training (e.g., DataParallel or DDP)."
    )

    d_devices: List[int] = Field(
        default_factory=lambda: [0],
        description="List of GPU device IDs (e.g., [0,1,2])."
    )

    d_use_amp: bool = Field(
        False,
        description="Enable automatic mixed precision (AMP) for faster training on supported GPUs."
    )

    d_plot_results: bool = Field(
        False,
        description="Enable plotting."
    )

    # --------------------------------
    # Allow CLI-style input: "0,1,2"
    # --------------------------------
    @field_validator("d_devices", mode="before")
    def parse_devices(cls, v):
        if isinstance(v, str):
            try:
                return [int(x.strip()) for x in v.split(",")]
            except ValueError:
                raise ValueError("d_devices must be a comma-separated list of integers.")
        return v

    # --------------------------------
    # Semantic validation
    # --------------------------------
    @model_validator(mode="after")
    def validate_gpu_config(self):
        # Multi-GPU requires at least 2 devices
        if self.d_use_multi_gpu:
            if len(self.d_devices) < 2:
                raise ValueError(
                    "d_devices must contain at least 2 GPU IDs when d_use_multi_gpu is True."
                )

        # Ensure primary GPU is part of device list
        if self.d_gpu not in self.d_devices:
            raise ValueError(
                "d_gpu must be one of the IDs in d_devices."
            )

        return self    
    
    # -------------------------
    # Cross-field validation
    # -------------------------
    @model_validator(mode="after")
    def validate_lengths(self):
        # label_len should not exceed seq_len
        if self.d_label_len > self.d_seq_len:
            raise ValueError("d_label_len must be <= d_seq_len.")

        return self