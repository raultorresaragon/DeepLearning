# DeepLearning: Compare/Contrast InstructPix2Pix and DreamBooth

## DreamBooth: Introduction
DreamBooth is a fine-tuning mechanism. It involves passing 5 _instance images_ a pre-trained text-to-image model (like Stable Diffusion). These instance images are supposed to pictures of your pet, or of your best friend, or of your favorite water bottle--something personal to you. Using the DreamBooth mechanism, the pre-trained model will learn what your subject looks like and then will be able to put that subject into different contexts for example, or change certain properties of the subject. The general idea is, "I want to generate an image of my dog in Antarctica" or "I want to generate an image of my dog wearing sunglasses".

Part of the way DreamBooth accomplishes this is by incorporating both reconstruction loss and class prior loss into the total loss. Reconstruction loss is a staple of diffusion models, with the idea being in the reverse pass to predict how much noise was added to the image at each time step t in the forward pass. Class prior loss is about ensuring that the model can produce images of the specifc subject along with diverse images of the class the subject belongs to. This is accomplished by comparing the outputs of the pre-trained and fine-tuned models. For example, if you were fine-tuning on your specific dog, the class prior loss would be the difference in outputs after passsing "a dog" to both the pre-trained and fine-tuned models. Practically speaking, one would want to generate ~200 class images on the pre-trained model.

Our DreamBooth implementation draws heavily on an implementation from the diffusers team from HuggingFace. That implementation can be found here: https://github.com/huggingface/community-events/tree/main/keras-dreambooth-sprint

To learn more about the specifics of DreamBooth, please see the original paper by Ruiz et al. (2022): https://dreambooth.github.io/

## DreamBooth: Our Implementation
Our implementation work with Google Colab and Google Drive. Class images and instance images should be saved to Google Drive folders. In the DreamBooth script, you will mount your Google Drive and load the paths to the image files.

Class images can be generated using "class_prior_generator_V2.ipynb". In this script, change the class prompt to whatever class your intance images belong to. 

We have included six DreamBooth models. Each file is essentially the same, except pointing to a different set of instance and class images. The way to use this git is to duplicate one of the "DreamBoothV2--....ipynb" files and change paths to the images file. Then, at inference, make sure to use the unique identifier in the prompt. We have it set to "sks". 

During trainig, losses will be reported. Our average loss across our models was ~0.12. You can modify the hyperparameters in the Trainer Initialization if you would like the loss to be improved further.
