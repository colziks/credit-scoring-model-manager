% MISiS article template — оформление по Article_1.docx (IEEE-style, 2 columns)
%
% Copyright (C) 2026 sadekovlar
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3c
% of this license or (at your option) any later version.
% The latest version of this license is in
%   https://www.latex-project.org/lppl.txt
% and version 1.3c or later is part of all distributions of LaTeX
% version 2008/05/04 or later.
%
% This work has the LPPL maintenance status `maintained'.
% The Current Maintainer of this work is sadekovlar.
%
% This work consists of the file misisarticle.cls.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{misisarticle}[2026/08/23 v1.1 MISiS article template]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt, twocolumn, a4paper]{article}

\author{}
\date{}

\RequirePackage[
  top=1.9cm,
  bottom=2.54cm,
  left=1.57cm,
  right=1.57cm,
  headheight=14pt,
  headsep=0.5cm,
  footskip=1cm
]{geometry}
\setlength{\columnsep}{0.635cm}

\RequirePackage{fontspec}
\RequirePackage{polyglossia}
\setdefaultlanguage{russian}
\setotherlanguage{english}

\RequirePackage{etoolbox}
\RequirePackage{microtype}
\RequirePackage{graphicx}
\RequirePackage{amsmath, amssymb}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{indentfirst}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage[hyphens]{url}
\RequirePackage{xurl}
\RequirePackage{hyperref}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  pdfborder={0 0 0}
}

\IfFontExistsTF{Times New Roman}{%
  \setmainfont{Times New Roman}[
    Ligatures=TeX,
    BoldFont={Times New Roman Bold},
    ItalicFont={Times New Roman Italic},
    BoldItalicFont={Times New Roman Bold Italic},
  ]%
  \newfontfamily\cyrillicfont{Times New Roman}[
    Ligatures=TeX,
    BoldFont={Times New Roman Bold},
    ItalicFont={Times New Roman Italic},
    BoldItalicFont={Times New Roman Bold Italic},
  ]%
  \newfontfamily\cyrillicfontsf{Times New Roman}[
    BoldFont={Times New Roman Bold},
    ItalicFont={Times New Roman Italic},
    BoldItalicFont={Times New Roman Bold Italic},
  ]%
  \newfontfamily\cyrillicfonttt{Courier New}%
}{%
  \setmainfont{TeX Gyre Termes}[Ligatures=TeX]%
  \newfontfamily\cyrillicfont{TeX Gyre Termes}[Ligatures=TeX]%
  \newfontfamily\cyrillicfontsf{TeX Gyre Termes}%
  \newfontfamily\cyrillicfonttt{TeX Gyre Cursor}%
}

\setlength{\parindent}{0.51cm}
\setlength{\parskip}{0pt plus 1pt minus 1pt}
\setstretch{1.0}

% Без переносов слов по всему документу
\hyphenpenalty=10000
\exhyphenpenalty=10000
\tolerance=9999
\emergencystretch=3em

\setlist[itemize]{
  label=\textbullet,
  leftmargin=0.9cm,
  labelsep=0.25cm,
  topsep=0pt,
  partopsep=0pt,
  itemsep=0pt,
  parsep=0pt
}

\newcommand{\@grnticodes}{}
\newcommand{\@asjccodes}{}
\newcommand{\@oecdcodes}{}
\newcommand{\grnti}[1]{\gdef\@grnticodes{#1}}
\newcommand{\asjc}[1]{\gdef\@asjccodes{#1}}
\newcommand{\oecd}[1]{\gdef\@oecdcodes{#1}}

\newcommand{\@classificationblock}{%
  {\fontsize{12}{14}\selectfont
    \raggedright
    \ifx\@grnticodes\empty\else
      ГРНТИ~\@grnticodes\par
    \fi
    \ifx\@asjccodes\empty\else
      ASJC~\@asjccodes\par
    \fi
    \ifx\@oecdcodes\empty\else
      OECD~\@oecdcodes\par
    \fi
    \vspace{0.3\baselineskip}%
  }%
}

\makeatletter
\newcounter{misis@authorcount}
\newcommand{\misis@authorsleft}{}
\newcommand{\misis@authorsright}{}

\newcolumntype{m}[1]{>{\centering\arraybackslash}p{#1}}

\newcommand{\misis@authorblock}[4]{%
  {\fontsize{11}{13}\selectfont
    #1\par
    #2\par
    #3\par
    \href{mailto:#4}{#4}\par
    \vspace{0.35\baselineskip}}%
}

\newcommand{\misisauthor}[4]{%
  \stepcounter{misis@authorcount}%
  \ifnum\value{misis@authorcount}=1
    \g@addto@macro\misis@authorsleft{\misis@authorblock{#1}{#2}{#3}{#4}}%
  \else
    \g@addto@macro\misis@authorsright{\misis@authorblock{#1}{#2}{#3}{#4}}%
  \fi
}

\newcommand{\@authorcolumns}{%
  \noindent
  \begin{tabular}{@{}m{0.44\textwidth}@{}m{0.04\textwidth}@{}m{0.44\textwidth}@{}}%
    \misis@authorsleft &
    &
    \misis@authorsright
  \end{tabular}%
}

\let\abstract\relax
\let\endabstract\relax
\newcommand{\@abstracttext}{}
\newcommand{\@keywordstext}{}
\newcommand{\setabstract}[1]{\gdef\@abstracttext{#1}}
\newcommand{\setkeywords}[1]{\gdef\@keywordstext{#1}}
\newcommand{\abstract}[1]{\setabstract{#1}}
\newcommand{\keywords}[1]{\setkeywords{#1}}

\renewcommand{\maketitle}{%
  \twocolumn[%
    \thispagestyle{empty}%
    \@classificationblock
    \begin{center}
      {\fontsize{24}{28}\selectfont \@title\par}
      \vspace{1\baselineskip}
    \end{center}
    \@authorcolumns
    \vspace{0.2\baselineskip}%
  ]%
  \ifx\@abstracttext\empty\else
    {\parindent=0.51cm
     \textbf{Аннотация — \@abstracttext}\par
     \vspace{0.25\baselineskip}}%
  \fi
  \ifx\@keywordstext\empty\else
    {\parindent=0.48cm
     \textbf{Ключевые слова — \@keywordstext}\par
     \vspace{0.25\baselineskip}}%
  \fi
}
\makeatother

\setcounter{secnumdepth}{2}
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}

\titleformat{\section}
  {\normalfont\scshape\centering}
  {\thesection.}
  {0.4em}
  {\MakeUppercase}
\titlespacing*{\section}{0pt}{8pt}{4pt}

\titleformat{\subsection}
  {\normalfont\itshape\raggedright}
  {\thesubsection.}
  {0.4em}
  {}
\titlespacing*{\subsection}{0pt}{6pt}{3pt}

\titleformat{\subsubsection}
  {\normalfont\itshape\raggedright}
  {}
  {0pt}
  {}
\titlespacing*{\subsubsection}{0pt}{4pt}{2pt}

\addto\captionsrussian{%
  \renewcommand{\figurename}{Рис.}%
  \renewcommand{\tablename}{Таблица}%
}

\renewcommand{\thetable}{\Roman{table}}

\captionsetup[figure]{
  font={footnotesize},
  labelfont={footnotesize},
  labelsep=period,
  skip=4pt,
  justification=centering
}

\captionsetup[table]{
  font={footnotesize},
  labelformat=simple,
  labelfont={footnotesize},
  textfont={footnotesize},
  labelsep=period,
  skip=6pt,
  position=above,
  justification=centering
}
\DeclareCaptionLabelFormat{misistable}{\scshape #1~#2}
\captionsetup[table]{labelformat=misistable}

\AtBeginEnvironment{table}{\small}

\setlength{\abovedisplayskip}{6pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{6pt plus 2pt minus 2pt}

\renewcommand{\refname}{ЛИТЕРАТУРА}

% Список литературы: заголовок по центру, [N] и текст на одной строке
\makeatletter
\renewenvironment{thebibliography}[1]{%
  \par\vspace{0.4\baselineskip}%
  \begin{center}%
    {\normalfont\scshape\MakeUppercase{\refname}}%
  \end{center}%
  \vspace{0.2\baselineskip}%
  \small
  \list{\@biblabel{\@arabic\c@enumiv}}%
       {\settowidth\labelwidth{\@biblabel{#1}}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \labelsep=0.4em
        \itemsep=2pt plus 1pt minus 1pt
        \parsep=0pt
        \listparindent=0pt
        \rightmargin=0pt
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\@arabic\c@enumiv}%
       }%
  \sloppy
  \clubpenalty4000
  \@clubpenalty\clubpenalty
}{%
  \endlist
}
\makeatother

\endinput
