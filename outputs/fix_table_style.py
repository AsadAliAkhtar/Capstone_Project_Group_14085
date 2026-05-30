"""
fix_table_style.py
==================
Post-render script: patches the 'Table' style in a Quarto-rendered .docx
to add borders, header shading, and row banding.

WHY THIS IS NEEDED
------------------
Quarto/Pandoc carries the Table style definition from its own internal
template rather than from your reference-doc. So even if custom-reference.docx
has a nicely styled 'Table' entry, the rendered output gets the plain
borderless definition. This script fixes that by directly editing the
styles.xml inside the output .docx after every render.

USAGE
-----
Run manually after quarto render:
  python fix_table_style.py report.docx

Or add to _quarto.yml as a post-render hook:
  post-render:
    - python fix_table_style.py report.docx

WHAT IT CHANGES
---------------
Inside the 'Table' style (w:styleId="Table"):
  1. Removes <w:semiHidden/> so the style is visible in Word's panel
  2. Adds outer borders: thick dark-gray single lines on all four sides
  3. Adds inner borders: thin light-gray lines between cells
  4. Adds header row (firstRow): bold white text on dark-blue background
  5. Adds banded rows (band1Horz): light-blue tint on alternating rows
  6. Sets comfortable cell padding (80/120 dxa top-bottom/left-right)

CUSTOMIZATION
-------------
Edit the constants below to match your document's color scheme.
Colors are in hex RGB without the # prefix.
Border sizes are in eighths of a point (sz=8 → 1pt, sz=4 → 0.5pt).
"""

import sys
import zipfile
import shutil
import os
import re
from pathlib import Path

# ── Style constants — edit these to match your color scheme ──────────────────
BORDER_OUTER_COLOR  = "404040"   # dark gray outer border
BORDER_INNER_COLOR  = "AAAAAA"   # light gray inner grid lines
BORDER_OUTER_SZ     = "8"        # 1pt outer border
BORDER_INNER_SZ     = "4"        # 0.5pt inner border
HEADER_FILL         = "2E5090"   # dark blue header background
BAND_FILL           = "EEF3FB"   # very light blue banded rows
CELL_PAD_TB         = "80"       # top/bottom cell padding in dxa
CELL_PAD_LR         = "120"      # left/right cell padding in dxa


# ── Replacement Table style XML ───────────────────────────────────────────────
NEW_TABLE_STYLE = f"""  <w:style w:type="table" w:customStyle="1" w:styleId="Table">
    <w:name w:val="Table"/>
    <w:qFormat/>
    <w:tblPr>
      <w:tblInd w:w="0" w:type="dxa"/>
      <w:tblBorders>
        <w:top     w:val="single" w:sz="{BORDER_OUTER_SZ}" w:space="0" w:color="{BORDER_OUTER_COLOR}"/>
        <w:left    w:val="single" w:sz="{BORDER_OUTER_SZ}" w:space="0" w:color="{BORDER_OUTER_COLOR}"/>
        <w:bottom  w:val="single" w:sz="{BORDER_OUTER_SZ}" w:space="0" w:color="{BORDER_OUTER_COLOR}"/>
        <w:right   w:val="single" w:sz="{BORDER_OUTER_SZ}" w:space="0" w:color="{BORDER_OUTER_COLOR}"/>
        <w:insideH w:val="single" w:sz="{BORDER_INNER_SZ}" w:space="0" w:color="{BORDER_INNER_COLOR}"/>
        <w:insideV w:val="single" w:sz="{BORDER_INNER_SZ}" w:space="0" w:color="{BORDER_INNER_COLOR}"/>
      </w:tblBorders>
      <w:tblCellMar>
        <w:top    w:w="{CELL_PAD_TB}" w:type="dxa"/>
        <w:left   w:w="{CELL_PAD_LR}" w:type="dxa"/>
        <w:bottom w:w="{CELL_PAD_TB}" w:type="dxa"/>
        <w:right  w:w="{CELL_PAD_LR}" w:type="dxa"/>
      </w:tblCellMar>
    </w:tblPr>
    <w:tblStylePr w:type="firstRow">
      <w:rPr>
        <w:b/>
        <w:color w:val="FFFFFF"/>
      </w:rPr>
      <w:tblPr/>
      <w:trPr>
        <w:tblHeader/>
      </w:trPr>
      <w:tcPr>
        <w:shd w:val="clear" w:color="auto" w:fill="{HEADER_FILL}"/>
        <w:vAlign w:val="center"/>
      </w:tcPr>
    </w:tblStylePr>
    <w:tblStylePr w:type="band1Horz">
      <w:tcPr>
        <w:shd w:val="clear" w:color="auto" w:fill="{BAND_FILL}"/>
      </w:tcPr>
    </w:tblStylePr>
  </w:style>"""


def patch_table_style(docx_path: str) -> None:
    """
    Patch the Table style in a .docx file in-place.

    Steps
    -----
    1. Read styles.xml from the docx zip
    2. Replace the entire <w:style ... w:styleId="Table"> block
    3. Write the modified styles.xml back into the zip
    """
    docx_path = Path(docx_path)
    if not docx_path.exists():
        print(f"Error: file not found: {docx_path}")
        sys.exit(1)

    # Work on a temp copy to be safe
    tmp_path = docx_path.with_suffix(".tmp.docx")
    shutil.copy2(docx_path, tmp_path)

    try:
        # Read all files from the zip
        with zipfile.ZipFile(tmp_path, 'r') as zin:
            names    = zin.namelist()
            contents = {name: zin.read(name) for name in names}

        # Patch styles.xml
        styles_xml = contents['word/styles.xml'].decode('utf-8')
        original   = styles_xml

        # Match the entire Table style block using a regex
        # Matches from <w:style ... w:styleId="Table"> to its closing </w:style>
        pattern = re.compile(
            r'<w:style\s[^>]*w:styleId="Table"[^>]*>.*?</w:style>',
            re.DOTALL
        )

        if not pattern.search(styles_xml):
            print("Warning: 'Table' style block not found in styles.xml.")
            print("Appending new style before </w:styles>.")
            styles_xml = styles_xml.replace(
                '</w:styles>',
                NEW_TABLE_STYLE + '\n</w:styles>'
            )
        else:
            styles_xml = pattern.sub(NEW_TABLE_STYLE, styles_xml)

        if styles_xml == original:
            print("No changes made — Table style already matches target.")
            tmp_path.unlink()
            return

        contents['word/styles.xml'] = styles_xml.encode('utf-8')

        # Write patched zip back to original path
        with zipfile.ZipFile(docx_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                zout.writestr(name, contents[name])

        tmp_path.unlink()
        print(f"✓ Table style patched in: {docx_path}")

    except Exception as e:
        # Restore original on failure
        shutil.copy2(tmp_path, docx_path)
        tmp_path.unlink()
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_table_style.py <output.docx>")
        sys.exit(1)
    patch_table_style(sys.argv[1])
