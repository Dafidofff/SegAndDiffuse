from GroundingDINO.groundingdino.util.inference import annotate, predict


def detect_bounding_box(image, image_source, text_prompt, model, box_threshold: float = 0.3, text_threshold: float = 0.25, device: str = 'cpu'):
    # Predict bounding box
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
    return annotated_frame, boxes