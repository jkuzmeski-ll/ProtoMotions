# Project working agreements

## Commit cadence

Commit incrementally as work progresses — don't leave large amounts of
uncommitted work sitting in the working tree. Prefer small, focused commits at
each meaningful checkpoint (a working fix, a passing test run, a completed
sub-step) so we always have a clean point to roll back to instead of one huge
blob.

- After a change is working and validated, commit it before moving on.
- Before a risky or exploratory change, commit the known-good state first.
- Keep commits scoped to one logical change where practical.
- Follow the commit-message style in the user's personal `AGENTS.md`
  (imperative subject <=50 chars, capitalized, no trailing punctuation; body
  wrapped at 72 only when it adds useful information).
- Committing is pre-authorized; no need to ask each time. Do not push, create
  branches, or rewrite history unless asked.

## Windows note

`2>nul` in shell commands creates a stray file named `nul` that breaks
`git add`. Avoid it; if it appears, `rm -f ./nul` before staging.
