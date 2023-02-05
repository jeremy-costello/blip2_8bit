import os
import gc
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


def main():
    folder_name = "images"

    name = "blip2_opt"
    model_type = "caption_coco_opt6.7b"
    load_in_8bit = True

    num_beams = 5
    length_penalty = 2.0
    repetition_penalty = 5.0

    captioner = Captioner(name, model_type, load_in_8bit, num_beams,
                          length_penalty, repetition_penalty)

    for entry in os.scandir(folder_name):
        if entry.is_file():
            captioner.caption_image(entry.name)


class Captioner:
    def __init__(self, name, model_type, load_in_8bit, num_beams,
                 length_penalty, repetition_penalty):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # load in model on CPU to optionally enable int8 inference
        model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type,
                                                            is_eval=True, device="cpu")
        
        # bitsandbytes doesn't work on windows :(
        if load_in_8bit:
            import bitsandbytes as bnb
            from accelerate import init_empty_weights
            
            self.model = self.replace_8bit_linear(model)
        else:
            self.model = model
        
        self.vis_processors = vis_processors

        del model
        del vis_processors
        gc.collect()
        
        self.model.to(self.device)
    
    def replace_8bit_linear(self, model, threshold=6.0, module_to_not_convert="lm_head"):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_8bit_linear(module, threshold, module_to_not_convert)

            if isinstance(module, torch.nn.Linear) and name != module_to_not_convert:
                with init_empty_weights():
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=threshold,
                    )
        return model

    def get_caption(self, image_name):
        raw_image = Image.open(image_name).convert("RGB")

        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        caption = self.model.generate({"image": image},
                                      num_beams=self.num_beams,
                                      length_penalty=self.length_penalty,
                                      repetition_penalty=self.repetition_penalty)

        return caption
    
    def caption_image(self, image_name):
        caption = self.get_caption(image_name)
        
        txt_name = '.'.join(image_name.split('.')[:-1]) + '.txt'
        with open(txt_name, 'w') as f:
            f.write(caption)


if __name__ == "__main__":
    main()
