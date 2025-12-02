| Configuration                       | Daily Cost (\$)      | Discomfort (°C·h)   | \textbf{Switches/day}              | Reward   | Hardware Safety                  |
|:------------------------------------|:---------------------|:--------------------|:-----------------------------------|:---------|:---------------------------------|
| Standard DRL ($w_3=0$)              | $1.2845              | 3.12                | 127                                | -198.30  | \textcolor{red}{UNSAFE}          |
| Low Penalty ($w_3=5$)               | $1.3124              | 2.95                | 68                                 | -185.70  | \textcolor{orange}{MARGINAL}     |
| \textbf{Proposed PI-DRL ($w_3=10$)} | $1.3521              | 2.81                | \textbf{38}                        | -178.40  | \textcolor{green}{\textbf{SAFE}} |
| High Penalty ($w_3=20$)             | $1.4203              | 2.76                | 22                                 | -172.10  | \textcolor{green}{SAFE}          |
|                                     |                      |                     |                                    |          |                                  |
| \textbf{Trade-off Analysis}         | \textit{+5.3\% cost} | —                   | \textit{\textbf{-70.1\%} switches} | —        | \textbf{Worth it!}               |