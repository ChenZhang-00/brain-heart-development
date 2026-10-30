"""Display helpers for the guided dBHI notebook."""


def show_table(path, *, columns=None, head=None, formats=None, title=None):
    """Display a saved CSV in the notebook without altering it."""
    import pandas as pd
    from IPython.display import Markdown, display

    table = pd.read_csv(path)
    if columns is not None:
        table = table[[c for c in columns if c in table.columns]]
    if head is not None:
        table = table.head(head)
    if title:
        display(Markdown(f"**{title}**"))
    styled = table.style.hide(axis="index")
    if formats:
        styled = styled.format({k: v for k, v in formats.items() if k in table.columns})
    display(styled)
    return table


def show_figure(path, width=1000):
    """Display a saved PNG in the notebook."""
    from IPython.display import Image, display

    display(Image(filename=str(path), width=width))
