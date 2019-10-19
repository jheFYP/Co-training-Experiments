# Co-training-Experiments
Co-training requires classifiers/networks to select a proportion of pseudo-labelled samples to improve the other classifier/network. Ideally these samples should be the most confident. 

The co-teaching algorithm presented in the paper: "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels" by
Bo Han*, Quanming Yao*, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, Masashi Sugiyama can be found: https://github.com/bhanML/Co-teaching. It is inspired by co-training and the networks select the low-loss samples to update the weights of the other network. I am investigating methods to improve how confident samples are selected during co-teaching.
