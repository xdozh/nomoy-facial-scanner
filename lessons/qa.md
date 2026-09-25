# QA/GATE agent lessons

- Gate F v1 incorrectly passed even though the exact required default command `python demo.py --video samples/test.mp4` exited early on the pre-existing append-only log. A documented workaround or explicit alternate output command does not satisfy the binding default-command Definition of Done.
- Future Gate F certification must execute the exact default command start-to-finish in the delivered workspace without deleting or overwriting existing evidence, then verify newly produced output paths from that run.
