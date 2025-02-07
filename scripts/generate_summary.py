def generate_summary(video_path):
    # Initialize components
    processor = AVProcessor()
    scorer = AVScorer.load_from_checkpoint("checkpoints/best.ckpt")
    aggregator = ShotAggregator(threshold=0.5)
    
    # Process video
    visual_feats, audio_feats = processor.process_video(video_path)
    shot_boundaries = processor._detect_shots(video_path)
    
    # Convert to tensors
    visual_tensor = torch.FloatTensor(visual_feats)
    audio_tensor = torch.FloatTensor(audio_feats)
    
    # Get scores
    with torch.no_grad():
        frame_scores = scorer(visual_tensor, audio_tensor)
    
    # Aggregate scores to shots
    selected_shots = aggregator(frame_scores, shot_boundaries)
    
    # Create final summary
    create_video_summary(video_path, selected_shots)