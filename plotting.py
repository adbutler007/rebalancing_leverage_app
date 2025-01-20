# from lets_plot import *
# LetsPlot.setup_html()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.dates as mdates
import os
from datetime import timedelta
import matplotlib.font_manager as fm
import seaborn as sns
import logging

class MatPlotting:
    def __init__(self, df, file_type='png'):
        self.df = df.copy()
        print("Initialized MatPlotting with a copy of the DataFrame")
        
        # Handle DatetimeIndex
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()
            print("Reset DatetimeIndex to a column")
            if self.df.columns[0].lower() in ['index', 'date']:
                self.df.rename(columns={self.df.columns[0]: 'Date'}, inplace=True)
                print(f"Renamed column '{self.df.columns[0]}' to 'Date'")
        
        # Ensure 'Date' column is datetime type
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            print("'Date' column converted to datetime type")
        
        # Drop rows with any NaN values
        initial_shape = self.df.shape
        self.df = self.df.dropna()
        final_shape = self.df.shape
        print(f"Dropped NaN values: from {initial_shape} to {final_shape}")
        
        self.file_type = file_type
        self.figsize = (12, 5)
        self.color_palette = [
            '#323A46',  # Dark gray/blue (preserved)
            '#3A6A9C',  # Medium blue (preserved)
            '#95E885',  # Light green (preserved)
            '#008EB8',  # Cyan blue (preserved)
            '#14CFA6',  # Turquoise (preserved)
            '#FF7E67',  # Coral
            '#6C5B7B',  # Dusty purple
            '#F8B195',  # Peach
            '#C06C84'   # Rose
        ]
        print(f"Set file type to '{self.file_type}' and figure size to {self.figsize}")
        
        # Add DM Sans font
        font_dir = 'resolve_plotting/resolve_plotting/fonts/DM-sans/'
        font_files = [
            os.path.join(font_dir, f) for f in ['DMSans-Regular.ttf', 
                                               'DMSans-Medium.ttf', 
                                               'DMSans-Bold.ttf']
        ]
        for font_file in font_files:
            if os.path.exists(font_file):
                fm.fontManager.addfont(font_file)
                print(f"Added font from '{font_file}'")
            else:
                print(f"Font file '{font_file}' does not exist")
        
        plt.rcParams['font.family'] = 'DM Sans'
        print("Set font family to 'DM Sans'")
    
    def log2_breaks(self, x):
        print(f"Calculating log2 breaks for x={x}")
        numeric_df = self.df.select_dtypes(include=[np.number])
        min_val_across_df = numeric_df.min().min()
        print(f"Minimum numeric value across DataFrame: {min_val_across_df}")
        
        if min_val_across_df >= 80:
            starting_point = 100
        elif 40 <= min_val_across_df < 80:
            starting_point = 50
        else:
            starting_point = 25
        print(f"Starting point set to {starting_point}")
        
        x_min = np.log2(2 * x / starting_point)
        x_min = np.ceil(x_min)
        exponents = np.arange(x_min + 1)
        breaks = starting_point * 2**exponents
        print(f"Calculated breaks: {breaks}")
        return breaks
    
    def dollar_log2_labels(self, breaks):
        labels = ["${:,}".format(int(round(b))) for b in breaks]
        print(f"Generated dollar labels: {labels}")
        return labels
    
    @staticmethod
    def dollar_formatter(x, p):
        return '${:,.0f}'.format(x)
    
    def plot(self, filename='equity_plot', plot_type='equity', target=None, lookback=252, legend_position='best'):
        print(f"\nPlotting type: '{plot_type}', filename: '{filename}'")
        print(f"DataFrame shape: {self.df.shape}")
        print(f"DataFrame columns: {self.df.columns.tolist()}")
        print(f"DataFrame head:\n{self.df.head()}")
        
        filename = filename.split('.')[0]
        
        try:
            if plot_type == 'bar':
                return self._plot_bar(filename, legend_position)
            elif plot_type == 'density':
                print("\nCalling _plot_density method")
                return self._plot_density(filename, legend_position)
            elif plot_type == 'drawdown':
                return self._plot_drawdown(filename, legend_position)
            elif plot_type == 'correlation':
                return self._plot_correlation(filename, target, lookback, legend_position)
            else:  # default to equity plot
                return self._plot_equity(filename, legend_position)
        except Exception as e:
            print(f"Error in plot: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_plot_style(self, ax):
        """Apply common style elements to the plot"""
        print("Setting up plot style")
        
        # Set background colors
        ax.set_facecolor('white')
        ax.figure.set_facecolor('white')
        print("Set white background")
        
        # Configure grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, 
                color='lightgray', alpha=0.5)
        print("Configured grid")
        
        # Style axis lines
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        print("Styled axis lines")
        
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', length=4, width=1)
        ax.tick_params(axis='both', which='minor', length=2, width=1)
        print("Configured tick parameters")
        
        # Determine date range and set appropriate formatting
        date_range = self.df['Date'].max() - self.df['Date'].min()
        years_of_data = date_range.days / 365.25
        print(f"Years of data: {years_of_data:.1f}")
        
        if years_of_data > 5:
            date_format = '%Y'
            # Set year interval based on data range
            if years_of_data >= 21:
                year_interval = 2  # Every other year
                print("Using year format with 2-year intervals (≥21 years)")
            else:
                year_interval = 1  # Every year
                print("Using year format with 1-year intervals (<21 years)")
            
            # Create custom year locator
            ax.xaxis.set_major_locator(mdates.YearLocator(base=year_interval))
        else:
            date_format = '%b-%Y'
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            print("Using month-year format for dates (≤5 years)")
        
        # Configure x-axis date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Set x-axis labels horizontal with no rotation
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        print("Set x-axis labels horizontal with no rotation")
        
        return ax
    
    def _setup_legend(self, ax, n_series, legend_position='best'):
        """Configure legend styling"""
        print(f"Setting up legend for {n_series} series at position '{legend_position}'")
        if n_series <= 5:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                      ncol=n_series, frameon=False, fontsize=12)
            print("Configured legend for <=5 series")
        else:
            ax.legend(loc=legend_position, frameon=False, fontsize=10)
            print("Configured legend for >5 series")
    
    def _plot_bar(self, filename, legend_position):
        """Create a bar plot with value labels above/below bars"""
        print("Starting bar plot creation")
        try:
            print(f"Input DataFrame shape: {self.df.shape}")
            print(f"Input DataFrame index: {self.df.index.name}")
            
            # Create figure and axis with white background
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
            print("Initialized figure and axis")
            
            # Apply plot style without date formatting
            self._setup_plot_style_no_date(ax)
            
            # Plot bars
            x = np.arange(len(self.df.index))
            # Calculate optimal bar width based on number of bars and columns
            n_bars = len(self.df.index)
            n_cols = len(self.df.columns)
            width = min(0.8 / n_cols, 0.35)  # Dynamically adjust width
            print(f"[DEBUG] Using bar width = {width:.2f} for {n_cols} columns and {n_bars} bars")
            
            # Get max and min values for setting y-axis limits with buffer
            max_val = self.df.max().max()
            min_val = self.df.min().min()
            
            # Plot bars and add value labels
            for i, column in enumerate(self.df.columns):
                bars = ax.bar(
                    x + i*width - width/2, 
                    self.df[column], 
                    width, 
                    label=column,
                    color=self.color_palette[i % len(self.color_palette)]
                )
                print(f"Plotted bars for {column}")

                for bar in bars:
                    height = bar.get_height()
                    value = height
                    if value >= 0:
                        va = 'bottom'
                        y_offset = max_val * 0.02
                    else:
                        va = 'top'
                        y_offset = min_val * 0.02
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        value + y_offset,
                        f'{value:.2%}',
                        ha='center',
                        va=va,
                        fontsize=10,
                        family='DM Sans'
                    )
                print(f"Added value labels for {column}")
            
            y_buffer = max(abs(max_val), abs(min_val)) * 0.1
            ax.set_ylim(min_val - y_buffer, max_val + y_buffer)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.df.index, rotation=0, fontsize=10, family='DM Sans')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            ax.tick_params(axis='y', labelsize=10)
            print("Customized plot formatting")
            
            self._setup_legend(ax, len(self.df.columns), legend_position)
            self._add_border_lines_bar(ax)
            
            plt.tight_layout()
            plot_path = f'{filename}.{self.file_type}'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved bar plot to {plot_path}")
            return plot_path
        
        except Exception as e:
            print(f"Error in _plot_bar: {str(e)}")
            raise
        finally:
            plt.close('all')
            print("Closed all figures")
    
    def _plot_equity(self, filename, legend_position):
        print("Starting equity plot creation")
        try:
            plot_df = self.df.copy()
            print("Created copy of DataFrame for equity plot")
            
            # Normalize to 100
            for col in plot_df.columns:
                if col != 'Date':
                    plot_df[col] = 100 * plot_df[col]
                    print(f"Normalized column '{col}' to 100-based")
            
            # Calculate plot limits
            max_val = plot_df.drop('Date', axis=1).max().max()
            min_val = plot_df.drop('Date', axis=1).min().min()
            print(f"Max value: {max_val}, Min value: {min_val}")
            
            # Adjust max_y based on the actual maximum value
            if max_val > 200:
                new_max_y = max_val * 2
                y_axis_min = min(80, min_val)
                print(f"Using log scale with max_y: {new_max_y}, min_y: {y_axis_min}")
            else:
                buffer = (max_val - min_val) * 0.1  # 10% buffer
                new_max_y = max_val + buffer
                y_axis_min = max(min_val - buffer, 0)  # Ensure we don't go below 0
                print(f"Using linear scale with max_y: {new_max_y}, min_y: {y_axis_min}")
            
            # Create figure with white background
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
            print("Initialized figure and axis")
            
            # Apply plot style
            self._setup_plot_style(ax)
            
            # Plot each series
            colors = self.color_palette * (len(plot_df.columns) // len(self.color_palette) + 1)
            for idx, column in enumerate(plot_df.columns):
                if column != 'Date':
                    ax.plot(plot_df['Date'], plot_df[column], 
                           label=column, linewidth=2,
                           color=colors[idx % len(colors)])
                    print(f"Plotted series '{column}' with color '{colors[idx % len(colors)]}'")
            
            # Y-axis scaling and formatting
            if max_val > 200:
                print("Applying log scale to y-axis")
                ax.set_yscale('log', base=2)
                breaks = self.log2_breaks(new_max_y)
                ax.set_yticks(breaks)
                ax.set_yticklabels(['${:,}'.format(int(b)) for b in breaks])
            else:
                print("Applying linear scale to y-axis")
                ax.yaxis.set_major_formatter(FuncFormatter(self.dollar_formatter))
            ax.set_ylim([y_axis_min, new_max_y])
            
            # Add reference line at 100
            ax.axhline(y=100, linestyle='--', color='black', alpha=0.5, linewidth=1)
            print("Added reference line at 100")
            
            # Set x-axis limits with 3-day padding
            min_date = plot_df['Date'].min()
            max_date = plot_df['Date'].max() + pd.Timedelta(days=3)
            ax.set_xlim([min_date, max_date])
            print(f"Set x-axis limits to [{min_date}, {max_date}]")
            
            # Add border lines and legend
            self._add_border_lines(ax, new_max_y, max_date)
            self._setup_legend(ax, len(plot_df.columns) - 1, legend_position)
            
            # Remove labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            print("Removed axis labels")
            
            # Save plot
            plt.tight_layout()
            plot_path = f'{filename}.{self.file_type}'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {plot_path}")
            plt.close()
            return plot_path
        
        except Exception as e:
            print(f"Error in _plot_equity: {str(e)}")
            raise
        finally:
            plt.close('all')
            print("Closed all figures")
    
    def _plot_drawdown(self, filename, legend_position):
        print("Starting drawdown plot creation")
        try:
            plot_df = self.df.copy()
            print("Created copy of DataFrame for drawdown plot")
            
            # Calculate drawdowns for each column except Date
            for col in plot_df.columns:
                if col != 'Date':
                    rolling_max = plot_df[col].expanding().max()
                    drawdown = (plot_df[col] - rolling_max) / rolling_max * 100
                    plot_df[col] = drawdown
                    print(f"Calculated drawdown for '{col}'")
            print("Calculated drawdowns for all applicable columns")
            
            # Create figure with white background
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
            print("Initialized figure and axis for drawdown plot")
            
            # Apply plot style
            self._setup_plot_style(ax)
            
            # Plot each series
            colors = self.color_palette * (len(plot_df.columns) // len(self.color_palette) + 1)
            for idx, column in enumerate(plot_df.columns):
                if column != 'Date':
                    ax.plot(plot_df['Date'], plot_df[column], label=column, linewidth=2,
                            color=colors[idx % len(colors)])
                    print(f"Plotted drawdown series '{column}' with color '{colors[idx % len(colors)]}'")
            
            # Configure legend with improved styling
            n_series = len(plot_df.columns) - 1
            self._setup_legend(ax, n_series, legend_position)
            
            # Y-axis formatting
            min_drawdown = plot_df.drop(columns='Date').min().min()
            y_limits = [min_drawdown - 5, 0]
            ax.set_ylim(y_limits)
            ax.yaxis.set_major_formatter(PercentFormatter())
            print(f"Set y-axis limits to {y_limits} and applied PercentFormatter")
            
            # X-axis limits and formatting
            min_date = plot_df['Date'].min()
            max_date = plot_df['Date'].max() + timedelta(days=21)
            ax.set_xlim([min_date, max_date])
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            print(f"Set x-axis limits to [{min_date}, {max_date}] and formatted x-axis")
            
            # Add black border lines at top and right
            ax.axhline(y=y_limits[0], linewidth=2, color='black')
            ax.axvline(x=max_date, linewidth=2, color='black')
            print("Added black border lines at top and right")
            
            # Reference line at 0
            ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.5)
            print("Added reference line at y=0")
            
            # Labels
            ax.set_ylabel("Drawdown")
            ax.set_xlabel("")
            print("Set y-axis label to 'Drawdown' and cleared x-axis label")
            
            # Save plot
            plt.tight_layout()
            plot_path = f'{filename}.{self.file_type}'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved drawdown plot to '{plot_path}'")
            plt.close()
            print("Drawdown plot creation complete")
            return plot_path
                
        except Exception as e:
            print(f"Error in _plot_drawdown: {str(e)}")
            raise
        finally:
            plt.close('all')
            print("Closed all figures for drawdown plot")
    
    def _plot_correlation(self, filename, target, lookback, legend_position):
        print("Starting correlation plot creation")
        try:
            plot_df = self.df.copy()
            print("Created copy of DataFrame for correlation plot")
            
            # Create figure with white background
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
            print("Initialized figure and axis for correlation plot")
            
            # Apply plot style (this will handle date formatting consistently)
            self._setup_plot_style(ax)
            
            # Plot each correlation series
            correlation_cols = [col for col in plot_df.columns if col != 'Date']
            colors = self.color_palette * (len(correlation_cols) // len(self.color_palette) + 1)
            
            for idx, column in enumerate(correlation_cols):
                ax.plot(plot_df['Date'], plot_df[column], label=column, linewidth=2,
                        color=colors[idx % len(colors)])
                print(f"Plotted correlation series '{column}' with color '{colors[idx % len(colors)]}'")
            
            # Configure legend
            self._setup_legend(ax, len(correlation_cols), legend_position)
            
            # Y-axis limits and formatting
            ax.set_ylim([-1, 1])
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
            print("Set y-axis limits to [-1, 1] with ticks every 0.2")
            
            # X-axis limits
            min_date = plot_df['Date'].min()
            max_date = plot_df['Date'].max() + pd.Timedelta(days=21)
            ax.set_xlim([min_date, max_date])
            print(f"Set x-axis limits to [{min_date}, {max_date}]")
            
            # Reference line at 0
            ax.axhline(y=0, linestyle='--', linewidth=1, color='black', alpha=0.5)
            print("Added reference line at y=0")
            
            # Labels
            ax.set_ylabel("Rolling Correlation")
            ax.set_xlabel("")
            print("Set y-axis label to 'Rolling Correlation' and cleared x-axis label")
            
            # Save plot
            plt.tight_layout()
            plot_path = f'{filename}.{self.file_type}'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved correlation plot to '{plot_path}'")
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Error in _plot_correlation: {str(e)}")
            raise
        finally:
            plt.close('all')
            print("Closed all figures for correlation plot")
    
    def _add_border_lines(self, ax, max_y, max_date):
        """Add thick black border lines at top and right of plot"""
        print("Adding border lines")
        try:
            # Add top border line
            ax.axhline(y=max_y, color='black', linewidth=2)
            print(f"Added top border line at y={max_y}")
            
            # Add right border line
            ax.axvline(x=max_date, color='black', linewidth=2)
            print(f"Added right border line at x={max_date}")
            
        except Exception as e:
            print(f"Error adding border lines: {str(e)}")
            raise

    def _setup_plot_style_no_date(self, ax):
        """Setup plot style without date-specific formatting"""
        print("Setting up plot style without date formatting")
        try:
            # Set white background
            ax.set_facecolor('white')
            print("Set white background")
            
            # Configure grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            print("Configured grid")
            
            # Style axis lines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            print("Styled axis lines")
            
            # Configure tick parameters
            ax.tick_params(axis='both', which='major', labelsize=10)
            print("Configured tick parameters")
            
        except Exception as e:
            print(f"Error in _setup_plot_style_no_date: {str(e)}")
            raise

    def _add_border_lines_bar(self, ax):
        """Add thick black border lines at top and right for bar plots"""
        print("Adding border lines for bar plot")
        try:
            # Get current axis limits
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            
            # Add top border line
            ax.axhline(y=y_max, color='black', linewidth=2)
            print(f"Added top border line at y={y_max}")
            
            # Add right border line
            ax.axvline(x=x_max, color='black', linewidth=2)
            print(f"Added right border line at x={x_max}")
            
        except Exception as e:
            print(f"Error adding border lines for bar plot: {str(e)}")
            raise

    def _plot_density(self, filename, legend_position):
        """Create a density plot with shaded regions"""
        print("\nStarting density plot creation")
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
            print("Created figure and axis")
            
            # Apply plot style
            self._setup_plot_style_no_date(ax)
            print("Applied plot style")
            
            x_index = self.df.index.values
            y_values = self.df['Density'].values
            
            # Calculate probabilities using trapezoidal integration
            dx = x_index[1] - x_index[0]
            mask_under = x_index < 0
            mask_over = x_index >= 0
            prob_under = np.trapezoid(y_values[mask_under], x_index[mask_under])
            prob_over = np.trapezoid(y_values[mask_over], x_index[mask_over])
            
            # Calculate mean values for each region
            mean_under = np.average(x_index[mask_under], weights=y_values[mask_under])
            mean_over = np.average(x_index[mask_over], weights=y_values[mask_over])
            
            # Plot regions with labels
            # Underperform region
            ax.fill_between(x_index[mask_under], y_values[mask_under], 
                           color=self.color_palette[0],
                           alpha=0.3,
                           label='Underperform')
            ax.plot(x_index[mask_under], y_values[mask_under],
                   color=self.color_palette[0],
                   linewidth=2)
            
            # Add underperform label
            ax.text(mean_under, np.max(y_values) * 0.7,
                    f'P(Underperform) = {prob_under:.1%}\nMean = {mean_under:.1%}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=self.color_palette[0],
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Outperform region
            ax.fill_between(x_index[mask_over], y_values[mask_over],
                           color=self.color_palette[1],
                           alpha=0.3,
                           label='Outperform')
            ax.plot(x_index[mask_over], y_values[mask_over],
                   color=self.color_palette[1],
                   linewidth=2)
            
            # Add outperform label
            ax.text(mean_over, np.max(y_values) * 0.7,
                    f'P(Outperform) = {prob_over:.1%}\nMean = {mean_over:.1%}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=self.color_palette[1],
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Format axes
            ax.set_xlabel('CAGR Difference (%)')
            ax.set_ylabel('Density')
            
            # Add legend
            self._setup_legend(ax, 2, legend_position)
            
            # Save plot
            plt.tight_layout()
            plot_path = f'{filename}.{self.file_type}'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            return plot_path
            
        except Exception as e:
            print(f"Error in _plot_density: {str(e)}")
            raise
        finally:
            plt.close('all')