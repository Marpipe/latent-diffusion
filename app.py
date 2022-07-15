import gradio as gr
import sys 
sys.path.append(".")
sys.path.append('./taming-transformers')
sys.path.append('./latent-diffusion')
from huggingface_hub import hf_hub_download
from run import process_run

model_path_e = hf_hub_download(repo_id="multimodalart/compvis-latent-diffusion-text2img-large", filename="txt2img-f8-large.ckpt")

#@title Import stuff
import sys

image = gr.outputs.Image(type="pil", label="Your result")
css = ".output-image{height: 528px !important} .output-carousel .output-image{height:272px !important} a{text-decoration: underline}"
iface = gr.Interface(
    fn=process_run, 
    inputs = [
        gr.inputs.Textbox(label="Prompt - try adding increments to your prompt such as 'oil on canvas', 'a painting', 'a book cover'",default="chalk pastel drawing of a dog wearing a funny hat"),
        gr.inputs.Slider(label="Steps - more steps can increase quality but will take longer to generate",default=45,maximum=50,minimum=1,step=1),
        gr.inputs.Radio(label="Width", choices=[32,64,128,256],default=256),
        gr.inputs.Radio(label="Height", choices=[32,64,128,256],default=256),
        gr.inputs.Slider(label="Images - How many images you wish to generate", default=2, step=1, minimum=1, maximum=4),
        gr.inputs.Slider(label="Diversity scale - How different from one another you wish the images to be",default=5.0, minimum=1.0, maximum=15.0),
        #gr.inputs.Slider(label="ETA - between 0 and 1. Lower values can provide better quality, higher values can be more diverse",default=0.0,minimum=0.0, maximum=1.0,step=0.1),
    ], 
    outputs=[image,gr.outputs.Carousel(label="Individual images",components=["image"]),gr.outputs.Textbox(label="Error")],
    css=css,
    title="Generate images from text with Latent Diffusion LAION-400M",
    description="<div>By typing a prompt and pressing submit you can generate images based on this prompt. <a href='https://github.com/CompVis/latent-diffusion' target='_blank'>Latent Diffusion</a> is a text-to-image model created by <a href='https://github.com/CompVis' target='_blank'>CompVis</a>, trained on the <a href='https://laion.ai/laion-400-open-dataset/'>LAION-400M dataset.</a><br>This UI to the model was assembled by <a style='color: rgb(245, 158, 11);font-weight:bold' href='https://twitter.com/multimodalart' target='_blank'>@multimodalart</a></div>",
    article="<h4 style='font-size: 110%;margin-top:.5em'>Biases acknowledgment</h4><div>Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exarcbates societal biases. According to the <a href='https://arxiv.org/abs/2112.10752' target='_blank'>Latent Diffusion paper</a>:<i> \"Deep learning modules tend to reproduce or exacerbate biases that are already present in the data\"</i>. The model was trained on an unfiltered version the LAION-400M dataset, which scrapped non-curated image-text-pairs from the internet (the exception being the the removal of illegal content) and is meant to be used for research purposes, such as this one. <a href='https://laion.ai/laion-400-open-dataset/' target='_blank'>You can read more on LAION's website</a></div><h4 style='font-size: 110%;margin-top:1em'>Who owns the images produced by this demo?</h4><div>Definetly not me! Probably you do. I say probably because the Copyright discussion about AI generated art is ongoing. So <a href='https://www.theverge.com/2022/2/21/22944335/us-copyright-office-reject-ai-generated-art-recent-entrance-to-paradise' target='_blank'>it may be the case that everything produced here falls automatically into the public domain</a>. But in any case it is either yours or is in the public domain.</div>"
)
iface.launch(enable_queue=True, server_name="0.0.0.0")
