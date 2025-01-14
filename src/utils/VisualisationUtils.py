import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

pd.options.plotting.backend = "plotly"

# Possible Options: "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
template = "seaborn"


def plot_column(df, column, type="hist", filename=None):
    if type == "bar":
        fig = df[column].plot.bar(template=template)
    elif type == "hist":
        fig = df[column].plot.hist(template=template)

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


def show_history(history, validation: bool = False):
    if validation:
        # Loss
        fig, axes = plt.subplots(figsize=(20, 5))
        # Train
        axes.plot(history.epoch, history.history['loss'], color='r', label='Train')
        axes.plot(history.epoch, history.history['val_loss'], color='b', label='Val')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()
        plt.savefig('loss.jpg')
        plt.show()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # loss
        axes[0].plot(history.epoch, history.history['loss'])
        axes[0].set_title('Train')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')


#Inputs:
#   tn - True Negatives
#   fn - False Negatives
#   fp - False Positives
#   tp - True Positives
def plt_confusion_matrix(tp, tn, fp, fn, total_background, total_foreground, filename=None):
    z = [[fp, tn], [tp, fn]]

    x = ['Lesion', 'Background']
    y = ['Background', 'Lesion']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    #ColorScale: aggrnyl, blugrn, brwnyl, Blues
    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='brwnyl')

    sub_title = "<b>Total Background Voxels: </b>" + str(total_background) + "<br><b>Total Lesion Voxels: </b>" + str(
        total_foreground)
    # add title
    fig.update_layout(title_text='<b>Confusion Matrix</b><br>', font_family="Courier New",
                      title_font_family="Times New Roman", title_font_size=16, title_subtitle_text=sub_title,
                      title_subtitle_font_family="Times New Roman", title_subtitle_font_size=12,
                      xaxis_title="Prediction", yaxis_title="Ground Truth"
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    #fig.add_annotation(dict(font=dict(color="black", size=14),x=0.5,y=-0.15,showarrow=False,text="Prediction",xref="paper",yref="paper"))

    # add custom yaxis title
    #fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.35, y=0.5, showarrow=False, text="Ground Truth", textangle=-90, xref="paper", yref="p
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=200, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


# Plot histogram as comparison between two classes/types.
# Input:    df -> DataFrame
#           col_name -> column to be plotted
#           comp_classes -> column for the classes
#           filename -> Path where the graph needs to be saved. Possible value: None, Path
def plt_comp_hist(df, col_name, comp_class, filename=None):
    fig = px.histogram(df, x=col_name, color=comp_class, template=template)

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


# Plot box plots
# Input:    df -> DataFrame
#           points -> 'outliers' - only outliers are displayed. 'all' - all points are plotted.
#           comp_classes -> column for the classes
#           filename -> Path where the graph needs to be saved. Possible value: None, Path
def plt_box_plots(df, points="outliers", color="None", filename=None):
    fig = px.box(df, x="Metric", y="Value", color=color, points=points, template=template)

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


def plt_violin_plot(df, filename=None):
    fig = px.violin(df, x="Metric", y="Value", color="Model",
                    violinmode='overlay',
                    template=template # draw violins on top of each other
                    # default violinmode is 'group' as in example above
                    )
    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)


def plt_scatter_plot(df, metric, filename=None):
    fig = px.scatter(df, x="Voxel Count", y=metric, color="Model", template=template, symbol="Model", width=1600, height=600)

    if filename is None:
        fig.show()
    else:
        fig.write_image(filename)
