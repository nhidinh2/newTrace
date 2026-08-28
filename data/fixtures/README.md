# Fixtures

Everything in this directory is **synthetic**. The publishers are invented
`*.example` domains, and every headline and sentence was written for this
repository. No real article text is redistributed, so the offline demo can be
committed and re-run without touching any publisher's terms or copyright.

Regenerate with `make fixtures` (deterministic, seed 549).

| File | Shape |
| --- | --- |
| `<topic>.gdelt.json` | GDELT DOC 2.0 `artlist` payload: url, title, seendate, domain, language. Titles only, as GDELT returns. |
| `<topic>.rss.xml` | RSS 2.0 feed carrying the descriptions GDELT's artlist mode omits. |

The corpus contains 135 articles across 41 labelled stories, including 20
verbatim syndicated copies so duplicate handling has something real to do.
Ground-truth story groupings live in `../labels/fixture_story_labels.csv`.
