# reference/

Vendored, read-only upstream documentation. **Do not edit anything in here.**

`LEGOEducation/` is a shallow clone of <https://github.com/LEGO/LEGOEducation>, the
authoritative reference for the `legoeducation` Python API used throughout this repo.
It is gitignored so upstream stays upstream; fetch or refresh it with:

```
python scripts/sync_lego_docs.py
```

Read it before writing any LEGO API call. Start with
`LEGOEducation/function_description.md` for signatures and `LEGOEducation/constants.md`
for constant names.
