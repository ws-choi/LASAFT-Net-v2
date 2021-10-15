## LASAFT-Net-v2

### Listen, Attend and Separate by Attentively aggregating Frequency Transformation

Woosung Choi, Yeong-Seok Jeong, Jinsung Kim, Jaehwa Chung, Soonyoung Jung, and Joshua D. Reiss

[Demonstration](https://ws-choi.github.io/LASAFT-Net-v2/) (under construction)

### Experimental Results

- Musdb 18

| model                   |     vocals    |     drums     |      bass     |     other     |      AVG      |
|-------------------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [Meta-TasNet](https://github.com/pfnet-research/meta-tasnet)  |      6.40     |      5.91     |      5.58     |      4.19     |      5.52     |
| [AMSS-Net](https://github.com/ws-choi/AMSS-Net) |      6.78     |      5.92     |      5.10     |      4.51     |      5.58     |
| [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) |     7.33    |      5.68     | 5.63 | 4.87 |      5.88     |
| LASAFT-Net-v2 | 7.57 | 6.13 |      5.28     | 4.87 | 5.96 |

- MDX Challenge ([Leaderboard A](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/leaderboards?challenge_leaderboard_extra_id=868&challenge_round_id=886))

| model                   | model type |     vocals    |     drums     |      bass     |     other     |      AVG      |
|-------------------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [KUILAB-MDX-Net](https://github.com/kuielab/mdx-net/tree/Leaderboard_A) | dedicated (1 source/ 1 model) | 8.901 | 7.173 | 7.232 | 5.636 | 7.236 |
| [LaSAFT-Net-v1](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT) (light) | conditioned (4 sources/ 1 model)  |  7.275		 | 5.935	 | 5.823	 | 4.557	 | 5.897 |
| [LASAFT-Net-v2](https://github.com/ws-choi/LASAFT-Net-v2/tree/mdx-medium-v2-669) (light) | conditioned (4 sources/ 1 model) |  7.324	 | 5.976	 | 5.884 | 4.642 | 5.957 |
