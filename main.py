import line_detection as ld
import orientation as orn
import skeletonize as sk
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import pandas as pd


#install orientationpy with PIP


def find_line_orientation():
    """
    Find the largest contoured shape in an image to detect the filaments present.

    Parameters:
    data folder with tiff images (data_concentration_lipid.composition_septin.type_image.number)

    Returns:
    plots of the orientation for the detected filaments
    """
    # find filaments by detecting the edges
    ld.main()

    # find the orientation of the detected filaments
    orn.main()
    # orientation_summary = orn.main()
    # print(orientation_summary.keys())
    # sns.relplot(
    #     data=orientation_summary, x="theta", y="coherency",
    #     # kind="line",
    #     hue="septin_type",
    # )

    # plot the orientation

    # Check data types (on original DataFrame)
    # Perform aggregation for summary-level plotting
    #
    # df_clean_summary = (
    #     orientation_summary
    #     .assign(lipid_composition=lambda d: d.lipid_composition.astype('category'))
    #     .assign(septin_type=lambda d: d.septin_type.astype('category'))
    #     .groupby(['lipid_composition', 'septin_type'])
    #     .agg({'theta': ['mean', 'min', 'max']})
    #     .reset_index()
    #     )
    #
    # orientation_summary['weighted_theta'] = (orientation_summary['theta'] * orientation_summary['energy'])
    #
    # weighted_data = (orientation_summary.groupby(['lipid_composition', 'septin_type'])
    #     .agg(weighted_theta_mean=('weighted_theta', 'sum'), energy_sum = ('energy', 'sum'))
    #     .assign(weighted_theta=lambda d: d['weighted_theta_mean'] / d['energy_sum'])
    #     .reset_index())
    #
    # sns.kdeplot(data = orientation_summary, x = 'theta', weights = orientation_summary['energy'],
    #         hue = 'file_name',
    #         fill = False,
    #         alpha = 0.6)
    #
    #
    # plt.title('Weighted Theta Distribution')
    # plt.xlabel('Theta')
    # plt.ylabel('Density (Weighted by Energy)')
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    find_line_orientation()
