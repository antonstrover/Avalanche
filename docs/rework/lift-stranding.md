# Lift stranding rework

## Implemented behavior

The simulator keeps lift queues separate from onboard lift locations.
Failure handling runs before lift queue service during each movement tick.

One unavailable mask covers controller closures, weather closures, scheduled closures, and mechanical stops.
An unavailable lift cannot board or move a skier.
The lift also clears its fractional service residual.
The lift stores no service credit while it remains unavailable.

## Queue state flow

```text
QUEUE on lift
  -> lift becomes unavailable
NODE at the recorded source
  -> finite physical route exists -> reset the queue counter -> choose again
  -> no finite physical route -> add one tick to the queue counter
  -> timeout boundary reached -> STRANDED at the source node
```

The failure returns every queued skier to the recorded lift source.
The return clears the queue ticket and the queue source field.
The return preserves the cumulative waiting time.
The next node-choice event can select a finite alternative route.

The simulator uses the physical onward route check for the queue counter.
Any finite physical route resets the counter immediately.
The reset happens before the skier enters another edge.

## Onboard state flow

```text
ACTIVE on lift
  -> lift becomes unavailable -> freeze remaining travel -> add one blocked tick
  -> service recovers below timeout -> reset the onboard counter -> resume movement
  -> timeout boundary reached -> STRANDED on the lift
STRANDED on lift
  -> service recovers -> reset the onboard counter -> remain fixed and STRANDED
```

An unavailable lift freezes each onboard skier's remaining travel time.
The simulator records onboard blocked time in a separate counter.
Recovery resets that counter before a later failure can start another period.
Separate failures never combine their blocked seconds.

Both counters use the configured stranded timeout.
Both counters use the shared time epsilon.
The transition occurs on the first matching tick boundary.
All new stranding transitions commit together.

A stranded status remains sticky after recovery.
Only an active onboard skier resumes movement.
A stranded skier cannot board or move.

## Metrics and reports

The evaluator records queue-no-route and onboard blocked skier-seconds separately.
The exact population counters remain evaluator-only state.

The route sensor reports only aggregate blocked counts.
It groups queued reports by the public source node.
It groups onboard reports by the public lift edge.

Each report arrives after one control interval.
The sensor applies relative uniform noise from minus five percent to plus five percent.
It applies the noise before `numpy.rint`.
It clips noisy counts at zero.
It applies one-percent missingness to each report element.
It never reports an exact blocked duration.

## Evidence

The failure schedule tests cover exact starts, exact endings, queue returns, rerouting, timeouts, and sticky recovery.
The stranded tests cover both counters, every closure source, immediate resets, and separate failure periods.
The lift invariants cover residual resets, normal throughput, capacity, FIFO order, and stranded boarding exclusion.
The population invariants cover simultaneous queue returns, stable counts, and valid locations.
The determinism tests replay failure and recovery transitions with the same seed.

## Exclusions

The simulator does not model rescue, lift downloading, or injury transitions.
The `INJURED` status remains unused.
This change adds no hard node-capacity mechanism.
