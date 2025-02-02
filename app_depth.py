import gradio as gr
from model import Model
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


def create_demo(model: Model):
    examples = [
        ["__assets__/depth_videos/butterfly.mp4",
         "white butterfly, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/deer.mp4",
         "oil painting of a deer, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/fox.mp4",
         "wild red fox is walking on the grass, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/girl_dancing.mp4",
         "oil painting of a girl dancing close-up, masterpiece, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/girl_turning.mp4",
         "oil painting of a beautiful girl, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/halloween.mp4",
         "beautiful girl halloween style, a high-quality, detailed, and professional photo"],
        ["__assets__/depth_videos/santa.mp4",
         "a santa claus, a high-quality, detailed, and professional photo"],
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Text and Depth Conditional Video Generation')

        with gr.Row():
            with gr.Column():
                with gr.Tab("Upload Video"):
                    input_video = gr.Video(label="Input Video", source='upload',
                                           type='numpy', format="mp4", visible=True).style(height="auto")

                with gr.Tab("Record Video"):
                    recorded_video = gr.Video(label="Record Video", source='webcam',
                                              type='numpy', format="mp4", visible=True).style(height="auto")

            with gr.Column():
                result = gr.Video(label="Generated Video").style(height="auto")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')

            with gr.Column():
                gr.Markdown("### Advanced Options")

                watermark = gr.Radio(["Picsart AI Research", "Text2Video-Zero",
                                      "None"], label="None", value='None', visible=False)
                chunk_size = gr.Slider(
                    label="Chunk size", minimum=2, maximum=16, value=2, step=1, visible=not on_huggingspace,
                    info="Number of frames processed at once. Reduce for lower memory usage.")
                merging_ratio = gr.Slider(
                    label="Merging ratio", minimum=0.0, maximum=0.9, step=0.1, value=0.0,
                    visible=not on_huggingspace,
                    info="Ratio of how many tokens are merged. The higher the more compression (less memory and faster inference).")

        inputs = [
            input_video,
            prompt,
            chunk_size,
            watermark,
            merging_ratio,
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_controlnet_depth,
                    cache_examples=on_huggingspace,
                    run_on_click=False,
                    )

        run_button.click(fn=model.process_controlnet_depth,
                         inputs=[recorded_video, *inputs],
                         outputs=result, )
    return demo
