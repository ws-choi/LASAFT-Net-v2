LASAFT-Net-v2:
=============

**Woosung Choi, Yeong-Seok Jeong, Jinsung Kim, Jaehwa Chung, Soonyoung Jung, and Joshua D. Reiss**

Welcome to the tutorial webpage of LASAFT-Net-v2, which is a conditioned source separation model.

It outperforms our previous version, called [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) {cite:p}`lasaft`, equipping LASAFT block-v2.

While the existing method {cite:p}`lasaft` only cares about the symbolic relationships between the target source symbol and latent sources, ignoring audio content, our new approach also considers audio content with listening mechanisms.

Below is the experimental results of LASAFT-Net-V2.

## Experimental Results

### Musdb 18 (No extra dataset)

| model                   | type |     vocals    |     drums     |      bass     |     other     |      AVG      |
|-------------------------|:----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [Demucs](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)  | multi-head  |     6.84     |      6.86     |      7.01     |      4.42     |      6.28     |
| [D3Net](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)  | dedicated  |     7.24     |      7.01     |      5.25     |      4.53     |      6.01     |
| [Meta-TasNet](https://github.com/pfnet-research/meta-tasnet)  | conditioned  |      6.40     |      5.91     |      5.58     |      4.19     |      5.52     |
| [AMSS-Net](https://github.com/ws-choi/AMSS-Net) | conditioned  |      6.78     |      5.92     |      5.10     |      4.51     |      5.58     |
| [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) | conditioned  |     7.33    |      5.68     | 5.63 | 4.87 |      5.88     |
| LASAFT-Net-v2 | conditioned  | 7.57 | 6.13 |      5.28     | 4.87 | 5.96 |

### MDX Challenge ([Leaderboard A](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/leaderboards?challenge_leaderboard_extra_id=868&challenge_round_id=886))

| model                   | type |     vocals    |     drums     |      bass     |     other     |      AVG      |
|-------------------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [KUILAB-MDX-Net](https://github.com/kuielab/mdx-net/tree/Leaderboard_A) | dedicated  | 8.901 | 7.173 | 7.232 | 5.636 | 7.236 |
| [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) (light) | conditioned   |  7.275		 | 5.935	 | 5.823	 | 4.557	 | 5.897 |
| [LASAFT-Net-v2](https://github.com/ws-choi/LASAFT-Net-v2/tree/mdx-medium-v2-669) (light) | conditioned  |  7.324	 | 5.976	 | 5.884 | 4.642 | 5.957 |

In this tutorial, we explain how to use, train and evaluate our model.

