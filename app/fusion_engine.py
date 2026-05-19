def multimodal_profile_analysis(
    image_prediction,
    image_confidence,
    text_prediction,
    text_confidence
):

    image_score = (
        image_confidence/100
        if image_prediction=="fake"
        else 1-(image_confidence/100)
    )

    text_score = (
        text_confidence/100
        if text_prediction=="suspicious"
        else 1-(text_confidence/100)
    )

    final_score = (
        0.6*image_score
        +
        0.4*text_score
    )

    if final_score >= 0.5:
        verdict = "Synthetic Profile Likely"
    else:
        verdict = "Profile Appears Genuine"

    return {
        "verdict":verdict,
        "risk_score":round(final_score,2)
    }
