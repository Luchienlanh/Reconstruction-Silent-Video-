"""
main.py
=======
Entry point for the refactored modular project.
This script demonstrates how to assemble the modules for training.
"""

import os
import torch
from torch.utils.data import DataLoader

# Import refactored modules
from data.dataset import VNLipDatasetV2, collate_pad_v2
from models.encoders.factory import build_encoder, VisualLandmarkEncoderV2
from models.decoders.siren import TFiLMSIRENDecoder
from models.decoders.upsample import MelTemporalUpsampleDecoder
from training.curriculum import train_curriculum, CurriculumConfig, PhaseConfig
from spikingjelly.activation_based import functional

def main():
    print("=== Vietnamese Lip Reading to Mel-Spectrogram ===")
    
    # 1. Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_dir = "FullFrame_test"
    dataset_output = "Dataset_Output"
    force_full_frame_input = True
    
    if not os.path.isdir(data_dir):
        print(f"Warning: Data directory {data_dir} not found. Please ensure data is prepared.")
        return

    # 2. Dataset
    print("\n[1] Initializing Dataset V2...")
    dataset = VNLipDatasetV2(
        data_dir=data_dir,
        max_frames=30,
        random_crop=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=dataset_output,
        enable_fallback=True,
        force_full_frame=force_full_frame_input,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad_v2,
    )
    print(f"Dataset loaded: {len(dataset)} items. Batch size: 4")

    # 3. Model Encoders
    print("\n[2] Initializing Models...")
    # Base visual encoder (non-spiking or spiking)
    encoder_type = "non_snn" # or "snn"
    visual_encoder = build_encoder(encoder_type).to(device)
    
    # Fused encoder with landmarks
    encoder = VisualLandmarkEncoderV2(
        visual_encoder,
        num_landmark_points=dataset.landmark_num_points,
        z_dim=512,
    ).to(device)
    
    # 4. Model Decoders
    base_decoder = TFiLMSIRENDecoder(
        hidden_dim=256,
        out_dim=80,
        num_layers=4,
        use_conv=True,
        output_activation=None,
    ).to(device)
    
    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    print("Models initialized and moved to device.")

    # 5. Curriculum Training Config
    print("\n[3] Setting up Curriculum Config...")
    curriculum_cfg = CurriculumConfig(
        phases=[
            PhaseConfig(phase_id=1, name="F0 (Pitch)",    num_epochs=5,  lr=2e-4),
            PhaseConfig(phase_id=2, name="Voice Tone",     num_epochs=10, lr=1e-4),
            PhaseConfig(phase_id=3, name="Oscillation",    num_epochs=8,  lr=5e-5),
            PhaseConfig(phase_id=4, name="Energy",         num_epochs=5,  lr=3e-5),
        ],
        checkpoint_dir="checkpoints_modular",
        save_every=5,
        log_memory_every=50,
    )
    
    print("\nStarting Curriculum Training...")
    # NOTE: Uncomment to start training. 
    result = train_curriculum(
        encoder=encoder,
        decoder=decoder,
        dataloader=dataloader,
        device=device,
        curriculum_config=curriculum_cfg,
        reset_net_fn=functional.reset_net if encoder_type == "snn" else None,
    )
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
