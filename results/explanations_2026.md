# Why the model ranked 2026 this way

# sha256_t1=249bb1ac9a623070c0573e5f2f40d1536f0122d86ad5f33a91ce170363e64f31
# sha256_t2=f9af1fd20747536eb06752cf202d0626c0ed55bd7f66507935ca83da27ad9cc0
# git_commit=36b162e

These are the model's arithmetic, not causes. Every feature is a within-edition z-score, so a contribution is measured against the average 2026 nominee. The explanation is read-only: nothing here is used to tune or change the model.

## T1 (M1, in-window information)

- **#1 Ousmane Dembélé**: utility +2.74 (plain first-pick probability 23.7%). Pushing up, relative to the average nominee: big-stage output (UCL knockout, tournament, finals) (+0.84); other honors (league, cups, awards) (+0.60); prior Ballon d'Or standing (+0.43). Pulling down: being the defending winner (-0.11).
- **#2 Harry Kane**: utility +2.32 (plain first-pick probability 15.7%). Pushing up, relative to the average nominee: big-stage output (UCL knockout, tournament, finals) (+1.59); team strength (club rating, UCL and tournament stage) (+0.22); other honors (league, cups, awards) (+0.22). Pulling down: major titles (UCL / World Cup / continental) (-0.28).
- **#3 Lamine Yamal**: utility +2.27 (plain first-pick probability 14.8%). Pushing up, relative to the average nominee: baseline fame (pre-season Wikipedia attention) (+0.92); prior Ballon d'Or standing (+0.41); major titles (UCL / World Cup / continental) (+0.25). Pulling down: big-stage output (UCL knockout, tournament, finals) (-0.15).
- **#4 Achraf Hakimi**: utility +1.44 (plain first-pick probability 6.5%). Pushing up, relative to the average nominee: major titles (UCL / World Cup / continental) (+0.77); prior Ballon d'Or standing (+0.33); other honors (league, cups, awards) (+0.22). Pulling down: playing as a defender (-0.12); attacking output (npxG+xA per 90) (-0.06); big-stage output (UCL knockout, tournament, finals) (-0.05).
- **#5 Michael Olise**: utility +0.98 (plain first-pick probability 4.1%). Pushing up, relative to the average nominee: other honors (league, cups, awards) (+0.60); team strength (club rating, UCL and tournament stage) (+0.22); season attention beyond what performance explains (+0.22). Pulling down: major titles (UCL / World Cup / continental) (-0.28); prior Ballon d'Or standing (-0.15); baseline fame (pre-season Wikipedia attention) (-0.06).

Why #1 Ousmane Dembélé is above #2 Harry Kane in M1: big-stage output (UCL knockout, tournament, finals) (-0.75); major titles (UCL / World Cup / continental) (+0.53); other honors (league, cups, awards) (+0.39); prior Ballon d'Or standing (+0.24) (total gap +0.42 utility).

## T2 (M2, with post-window information)

- **#1 Ousmane Dembélé**: utility +2.97 (plain first-pick probability 28.6%). Pushing up, relative to the average nominee: big-stage output (UCL knockout, tournament, finals) (+0.81); other honors (league, cups, awards) (+0.59); prior Ballon d'Or standing (+0.42). Pulling down: being the defending winner (-0.12).
- **#2 Lamine Yamal**: utility +2.27 (plain first-pick probability 14.1%). Pushing up, relative to the average nominee: baseline fame (pre-season Wikipedia attention) (+0.89); prior Ballon d'Or standing (+0.40); major titles (UCL / World Cup / continental) (+0.24). Pulling down: big-stage output (UCL knockout, tournament, finals) (-0.14); post-window attention change (-0.06); attention spike in the last 14 days (-0.00).
- **#3 Harry Kane**: utility +2.21 (plain first-pick probability 13.4%). Pushing up, relative to the average nominee: big-stage output (UCL knockout, tournament, finals) (+1.53); team strength (club rating, UCL and tournament stage) (+0.23); other honors (league, cups, awards) (+0.22). Pulling down: major titles (UCL / World Cup / continental) (-0.27); post-window attention change (-0.01).
- **#4 Achraf Hakimi**: utility +1.37 (plain first-pick probability 5.7%). Pushing up, relative to the average nominee: major titles (UCL / World Cup / continental) (+0.75); prior Ballon d'Or standing (+0.32); other honors (league, cups, awards) (+0.22). Pulling down: post-window form (-0.14); playing as a defender (-0.13); big-stage output (UCL knockout, tournament, finals) (-0.04).
- **#5 Michael Olise**: utility +1.20 (plain first-pick probability 4.9%). Pushing up, relative to the average nominee: other honors (league, cups, awards) (+0.59); post-window form (+0.30); team strength (club rating, UCL and tournament stage) (+0.23). Pulling down: major titles (UCL / World Cup / continental) (-0.27); prior Ballon d'Or standing (-0.15); baseline fame (pre-season Wikipedia attention) (-0.06).

Why #1 Ousmane Dembélé is above #2 Lamine Yamal in M2: big-stage output (UCL knockout, tournament, finals) (+0.96); baseline fame (pre-season Wikipedia attention) (-0.79); other honors (league, cups, awards) (+0.38); being the defending winner (-0.12) (total gap +0.71 utility).

## Biggest movers between T1 and T2

- Kylian Mbappé: rank 9 -> 6; utility change +0.14, mainly post-window form (+0.27); attacking output (npxG+xA per 90) (-0.07); match ratings (-0.02).
- Bruno Fernandes: rank 28 -> 26; utility change -0.03, mainly match ratings (-0.03); team strength (club rating, UCL and tournament stage) (-0.02); attacking output (npxG+xA per 90) (-0.02).
- Ferran Torres: rank 15 -> 17; utility change -0.09, mainly post-window attention change (-0.15); post-window form (+0.08); attacking output (npxG+xA per 90) (-0.04).
- Rodri: rank 12 -> 14; utility change -0.10, mainly post-window form (-0.14); attacking output (npxG+xA per 90) (+0.04); big-stage output (UCL knockout, tournament, finals) (+0.01).
- Marc Cucurella: rank 25 -> 27; utility change -0.22, mainly post-window form (-0.24); post-window attention change (-0.08); match ratings (+0.05).
