set(groot,'defaultFigureColormap', turbo())

set(groot,'defaultLineLineWidth',1.8)
set(groot, 'DefaultAxesLineWidth', 1.6)
set(groot,'defaulttextinterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')
set(groot,'DefaultTextFontname', 'CMU Serif')
set(groot,'DefaultAxesFontName', 'CMU Serif')

set(groot, 'DefaultAxesFontSize', 20)
set(groot, 'defaultfigurecolor', 'white')
set(groot, 'defaultAxesColor', '#eceff4')
set(groot, 'defaultAxesGridColor', 'white')
set(groot, 'defaultAxesGridAlpha', 1.0)


newcolors = ["#BF616A" "#D08770" "#EBCB8B" "#A3BE8C" "#B48EAD"];
newcolors1 = ["#3B4252" "#4C566A" "#D8DEE9"];
newcolors2 = ["#E5E9F0" "#ECEFF4" "#8FBCBB"];
newcolors3 = ["#88C0D0" "#81A1C1" "#5E81AC"];

hardcolors = ["#0072BD" "#D95319" "#EDB120" "#7E2F8E" "#77AC30" "#4DBEEE" "#A2142F"];

plotcolors = [0.1803921568627451, 0.20392156862745098, 0.25098039215686274; %black
    0.7490196078431373, 0.3803921568627451, 0.41568627450980394; % red
    0.3686274509803922, 0.5058823529411764, 0.6745098039215687; 
    0.560784313725, 0.7372549019607844, 0.7333333333333333];

set(groot,'defaultAxesColorOrder',plotcolors)
