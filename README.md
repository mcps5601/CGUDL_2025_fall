# CGUDL_2025_fall

- This repository is the course materials for Deep Learning at Chang Gung University, in the academic year 114 (Fall).
- Instructor: [Ying-Jia Lin](https://yingjialin.org/about/)
- Course ID: AIM011 &amp; HDM006
- YouTube Playlist: [連結](https://youtube.com/playlist?list=PL0bwsTyVtLVziJuDQ5-l6w_ONUwfLKnQG&si=BK6PGof8G0zrj8Cx)
## Course Information
| Week | Theme | Slide | Code | Slido | Video | Practice |
| --- | --- | --- | --- | --- | --- | --- |
|1| Introduction to Deep Learning / Syllabus | [`.pdf`](./slides/intro_0903.pdf) [`.pptx`](./slides/intro_0903.pptx) |  | [`Slido`](https://app.sli.do/event/w95AaADjCS4sQHdmD93Rw4) |  [`Video1`](https://youtu.be/6JyufuR2Zsk) [`Video2`](https://youtu.be/xorhS3_K8Rg) [`Video3`](https://youtu.be/Kuv2VZcRxHU) |
|2|神經網路與梯度下降| [`.pdf`](./slides/nn_gd_0910.pdf) [`.pptx`](./slides/nn_gd_0910.pptx) | | [`Slido`](https://app.sli.do/event/vaWcY1tmNxZM4RVBwS7B2e) | [`Video1`](https://youtu.be/amEj1qnuDXg) [`Video2`](https://youtu.be/-FZ09HhVQr4) [`Video3`](https://youtu.be/2YnvLUJiv4I) |
|3|反向傳播法| [`.pdf`](./slides/backprop_0917.pdf) [`.pptx`](./slides/backprop_0917.pptx) [`HW1`](./homework/HW1.pdf) | [`Jupyter`](./code/jupyter_基本功能.ipynb) [`NumPy`](./code/numpy_基本功能.ipynb) | [`Slido`](https://app.sli.do/event/bbuHZY1x65qHqEKE1GpFeC) | [`Video1`](https://youtu.be/KIwWNTKpThY) [`Video2`](https://youtu.be/ONKBJaokbTQ) [`Video3`](https://youtu.be/F7TUS1qxZuQ)  | [`Quiz`](./quizzes/w3.md)|
|4|最佳化方法: SGD, Momentum, RMSProp, Adam| [`.pdf`](./slides/optimizers_0924.pdf) [`.pptx`](./slides/optimizers_0924.pptx) | [`PyTorch Basics`](./code/pytorch_基本功能.ipynb) | [`Slido`](https://app.sli.do/event/34TnFi6Hfe7tixdF2Z7oDV) |  [`Video1`](https://youtu.be/WDWgRISf9D4) [`Video2`](https://youtu.be/g83_jXzZHh0) [`Video3`](https://youtu.be/nJpkmWbteTU) |
|5|常見損失函數介紹| [`.pdf`](./slides/objectives_1001.pdf) [`.pptx`](./slides/objectives_1001.pptx) | [`PyTorch GD`](./code/pytorch_gd.ipynb) [`PyTorch Modeling`](./code/pytorch_mnist.ipynb) | [`Slido`](https://app.sli.do/event/r6bsGhmsqjXx3WLiJiGQdT) |  [`Video1`](https://youtu.be/QszuttVKb0w) [`Video2`](https://youtu.be/h6EWSECwiZs) [`PyTorch1`](https://youtu.be/NEBJqya2IDs) [`PyTorch2`](https://youtu.be/41r4nmeGitk) |
|6|卷積神經網路| [`.pdf`](./slides/cnn_1008.pdf) [`.pptx`](./slides/cnn_1008.pptx) [`HW2`](https://docs.google.com/presentation/d/106yrLbieNpOiUJ5WCxhmafv9RL7n6cY5x1uoUc_QLaM/edit?usp=sharing) | [`CNN PyTorch`](./code/cnn_pytorch/) [`HW2_sample`](./homework/hw2_sample.ipynb) | [`Slido`](https://app.sli.do/event/fqEtyRv9jmhGyLAZYE2pjC) |  [`Video1`](https://youtu.be/mby3fLOpuKQ) [`Video2`](https://youtu.be/fCjoayOIqug) [`Video3`](https://youtu.be/5xQ-2iFBwVE) |
|7|過擬合、正規化、模型訓練技巧、期末專案介紹| [`.pdf`](./slides/training_tips_1015.pdf) [`.pptx`](./slides/training_tips_1015.pptx) [`projects.pdf`](./slides/projects_1015.pdf) [`projects.pptx`](./slides/projects_1015.pptx) | [`model.eval()`](./code/model_eval_behavior.ipynb) | [`Slido`](https://app.sli.do/event/trLLdJv5TjT9ahwZTURqNG) |
|8|自然語言處理:RNN與序列建模| [`.pdf`](./slides/rnn_1022.pdf) [`.pptx`](./slides/rnn_1022.pptx) | [`NLP in PyTorch`](./code/NN_中文文本分類.ipynb) | [`Slido`](https://app.sli.do/event/6514xqxQa15XW2PKgHA8hY)
|9|期中考| |
|10|自注意力機制模型| [`.pdf`](./slides/transformers_1105.pdf) [`.pptx`](./slides/transformers_1105.pptx) [`HW3`](https://docs.google.com/presentation/d/117264PHlW0yOc2YZV5fJiGRhGer78S0hs72tZ2eSULY/edit?usp=sharing) | [`Transformer`](./code/NN_中文文本分類v2.ipynb) [`HW3_sample`](./homework/hw3_sample.ipynb) | [`Slido`](https://app.sli.do/event/uuJpcD24bAi7Qj7EzSgQCm) |
|11|Vision Transformers (ViT)、自監督式學習與預訓練模型| [`.pdf`]() [`.pptx`]() | [`ViT`](./code/vit/) |[`Slido`](https://app.sli.do/event/f4AyC4nV31kbBjuL3zEi6U)|
|12|大型語言模型| | [🤗 `Transformers`]() | 
|13|大模型時代如何有效率訓練模型?| |[🤗 `PEFT`]() [🤗 `Quantization`]() |
|14|圖神經網路|
|15|強化學習|
|16|小組實作成果報告|

