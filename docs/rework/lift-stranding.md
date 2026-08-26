# Lift stranding rework

## Current behavior

The simulator treats every lift as a one-way edge.
The edge carries skiers uphill only.

The route table contains one safe route set for each ability.
Beginners may use green and blue pistes.
Intermediate skiers may also use red pistes.
Advanced skiers may use every piste.

A lift has no piste grade.
Every ability may use a lift when a safe onward route exists.
The simulator rejects advice for a lift without that route.

Population sampling chooses an exit reachable from the sampled entrance.
The choice also uses the sampled ability.
The simulator rejects a sampled group when no safe exit exists.

A skier can still start at an unsafe mountain node in a direct scenario.
That skier waits at the node.
The existing timer can then mark the skier as stranded.

The small resort has one blue connector on its required route.
This connector keeps the synthetic fixture usable for beginners.
The medium resort keeps its realistic terrain gaps.

## Future rework

TODO: Model lift downloading and assisted recovery.

Add explicit download support to selected lifts.
Model the downward queue and the service capacity.
Record a rescue when no ordinary safe route exists.
Replace the temporary lift viability rejection when recovery becomes available.
Test the new movement, capacity, trace, and determinism behavior.
