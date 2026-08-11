# lambda_android (OBD2 / Speeduino live dashboard)

Android Studio project (`lambda_android/`), repo `rcaldas-com/obd2`.
Distinct from the `car` Next.js maintenance-tracking web app in a
separate repo — that one only references OBD2 as "integration in
progress" in older notes; this repo is that integration, now well past
initial and running on real hardware.

**What it does:** live dashboard reading lambda/O2 sensor data over USB
from an ELM327 adapter, plus (newer) live data read directly from a
Speeduino ECU over USB, merged into the same dashboard. A combined
`.msl` TunerStudio-format logger (manual start/stop via a UI button, not
automatic) and a GPS-based speed override.

**Recent work:** removed the old automatic CSV logger
(`startCsvLog`/`writeCsvLine`/`stopCsvLog` in `MainActivity.java`) — it
ran unconditionally on every ELM327 connection, wrote 4 columns that were
always empty (`stft1`/`stft2`/`rpm`/`timing`, dropped from `LambdaData`
in an earlier cleanup that prioritized lambda read rate), and did
synchronous file I/O on the UI thread on every graph tick. The two
columns it did cover (lambda banks 1/2) are already in the `.msl` logger
and the live graph, so nothing was lost.

Hardened the `.msl` logger (`MslLogger.java`) instead of the CSV one:
periodic `fsync` to survive a hard power cut (not every line at 10Hz —
too costly — batched every ~10 lines), an `ACTION_SHUTDOWN`
`BroadcastReceiver` registered the same way as the existing USB
permission/detach receivers, to close the log file cleanly if the system
announces a shutdown before `onDestroy()` would otherwise fire, and a
visible "REC" indicator on the dashboard/graph screens so an active
recording isn't only visible from Settings.

New file: `GpsSpeedProvider.java`.

**Key files:** `MainActivity.java`, `MslLogger.java`,
`SpeeduinoManager.java`, `DeviceRoleManager.java`, `Elm327Manager.java`,
`UsbSerialSession.java`.

**Status:** confirmed working by the user on real hardware before this
was committed. No open issues from this work as of the last session.
