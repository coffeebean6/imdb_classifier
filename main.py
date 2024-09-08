import gradio as gr
from infer import infer_sentence


def infer(review_text):
    pred = infer_sentence(review_text)
    emoji = '😊' if (pred == 1) else '😢'
    emoji_output = f"<div style='font-size: 3em; text-align: center;'>{emoji}</div>"
    return pred, emoji_output


with gr.Blocks() as demo:
    gr.Markdown("### Sentiment classifier")
    gr.HTML('<img src="https://sprcdn-assets.sprinklr.com/674/8b955864-7307-4d41-8ded-c194170f5305-2729152590.jpg" style="width:100%">')
    review_text = gr.Textbox(label="Input a user review", lines=5)
    infer_button = gr.Button("Determine if it's positive")
    infer_emoji = gr.Markdown()
    infer_output = gr.Textbox(label="Output")
    
    infer_button.click(infer, inputs=review_text, outputs=[infer_output, infer_emoji])
        
demo.launch()

