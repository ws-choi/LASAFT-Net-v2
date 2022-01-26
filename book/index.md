<!-- #region -->
(index)=

LASAFT-Net-v2
=============

**Woosung Choi, Yeong-Seok Jeong, Jinsung Kim, Jaehwa Chung, Soonyoung Jung, and Joshua D. Reiss**

Welcome to the tutorial webpage of LASAFT-Net-v2, which is a conditioned source separation model.

It outperforms our previous version, called [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) {cite:p}`lasaft`, equipping LASAFT block-v2.

While the existing method {cite:p}`lasaft` only cares about the symbolic relationships between the target source symbol and latent sources, ignoring audio content, our new approach also considers audio content with listening mechanisms.

Below is the experimental results of LASAFT-Net-V2.

## Experimental Results

### Musdb 18 (No extra dataset)

| model                   | conditioned? |     vocals    |     drums     |      bass     |     other     |      AVG      |
|-------------------------|:----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [Demucs](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)  | X |     6.84     |      6.86     |     **7.01**    |      4.42     |      **6.28**     |
| [D3Net](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)  | X |     7.24     |      **7.01**     |      5.25     |      4.53     |      6.01     |
| [Meta-TasNet](https://github.com/pfnet-research/meta-tasnet)  | O  |      6.40     |      5.91     |      5.58     |      4.19     |      5.52     |
| [AMSS-Net](https://github.com/ws-choi/AMSS-Net) | O  |      6.78     |      5.92     |      5.10     |      4.51     |      5.58     |
| [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) | O  |     7.33    |      5.68     | 5.63 | **4.87** |      5.88     |
| LASAFT-Net-v2 | O  | **7.57** | 6.13 |      5.28     | **4.87** | 5.96 |
| LASAFT-Net-v2 (updated) | O  | 7.43±0.09 | 6.23±0.05 | 5.28±0.19     | 4.89±0.05 | 5.99±0.03 |


### MDX Challenge ([Leaderboard A](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/leaderboards?challenge_leaderboard_extra_id=868&challenge_round_id=886))


| model                   | conditioned? |     vocals    |     drums     |      bass     |     other     |      Song      |
|-------------------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Demucs++ | X | 7.968 | **8.037**	 | **8.115** | 5.193 | **7.328** |
| [KUILAB-MDX-Net](https://github.com/kuielab/mdx-net/tree/Leaderboard_A) |X  | **8.901** | 7.173 | 7.232 | **5.636** | 7.236 |
| Kazane Team | X | 7.686 | 7.018 | 6.993 | 4.901 | 6.649 |
| [LASAFT-Net-v2.0](https://github.com/ws-choi/LASAFT-Net-v2/tree/mdx-medium-v2-669) | O  |  **7.354**	 | **5.996**	 | **5.894** | **4.595** | **5.960** |
| LaSAFT-Net-v1.2 | O   |  7.275		 | 5.935	 | 5.823	 | 4.557	 | 5.897 |
| Demucs48-HQ | X | 6.496	 | 6.509	 | 6.470 | 4.018 | 5.873 |
| LaSAFT-Net-v1.1 | O | 6.685  | 5.272     | 5.498 | 4.121 | 5.394 |
| XUMXPredictor |X |  6.341 | 5.807	| 5.615 | 3.722 | 5.372 | 
| UMXPredictor |X | 5.999	| 5.504 | 5.357 | 3.309 | 5.042 |

> LaSAFT-Net-v1.1 is also known as lightsaft-net

In this tutorial, we explain how to use, train and evaluate our model.

<!-- #endregion -->
