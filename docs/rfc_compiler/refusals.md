# refusals

The checker is the only place a program is refused. Once it admits a program, nothing after it refuses that program; an implementation that cannot run it has reached its own limit.

A refusal speaks in the program's terms. It names what the program wrote, where it wrote it, and the demand that failed. A part the program never determined is not drawn as a type; the refusal says which value left it open ("the length of `xs` is not settled yet").

The user's distinctions decide the structure. Where the user wrote three different things, they are told three different things, even when one check finds all three. Where the user sees one fact, the compiler keeps one path for it, even when it arises from different constructs. A new construct brings its own message, never its own tracking path.
