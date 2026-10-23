Sonic sources from waywardgeek/sonic commit b93885d, as vendored
by callebtc/sayit commit 37d49718bb58853945577a6845dce0cfde84c90d.

Sonic is Copyright 2010 Bill Cox and distributed under Apache-2.0; see LICENSE
and the copyright notices in sonic.c and sonic.h. It is statically compiled
into the service on Linux and Windows. No system Sonic installation is needed.

Local change on 2026-10-23: pitch-period comparisons use uint64_t instead of
unsigned long, which is only 32 bits on Windows. The quality=1 path operates
at full sample rate and can overflow the original cross-products. A 44.1 kHz
220 Hz tone at 3x reproduced the error with 32-bit arithmetic (243 Hz); the
64-bit implementation preserves pitch. The cross-platform DSP suite covers it.
