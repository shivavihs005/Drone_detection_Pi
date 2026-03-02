class FusionModule:
    def __init__(self):
        self.fusion_weight_vis = 0.6
        self.fusion_weight_aud = 0.4
        
    def fuse(self, vision_conf, audio_conf):
        """
        Simple weighted fusion logic
        """
        fusion_score = (self.fusion_weight_vis * vision_conf) + (self.fusion_weight_aud * audio_conf)
        fusion_score = min(max(fusion_score, 0.0), 1.0)
        
        # Threshold check
        is_detected = fusion_score > 0.75
        
        return {
            "vision": vision_conf,
            "audio": audio_conf,
            "fusion": fusion_score,
            "is_detected": is_detected
        }
