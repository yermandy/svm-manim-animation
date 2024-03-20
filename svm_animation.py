import numpy as np
from manim import *
from sklearn.datasets import make_classification
from sklearn.svm import SVC


class SVMAnimation(Scene):
    def construct(self):
        # Set random seed
        np.random.seed(42)

        # Generate random data for classification
        X, y = make_classification(n_samples=40, n_features=2, n_redundant=0, n_classes=2)

        # Create a NumberPlane to represent the feature space
        plane = NumberPlane(x_range=[-5, 5, 1], y_range=[-5, 5, 1], background_line_style={"stroke_opacity": 0.4})

        self.wait(0.5)
        self.play(Create(plane))

        # Add labels and legend
        x_label = plane.get_x_axis_label("x_1", direction=DOWN)
        y_label = plane.get_y_axis_label("x_2", direction=LEFT)

        self.play(
            Create(x_label),
            Create(y_label),
        )

        self.wait(0.2)

        # Plot data points on the NumberPlane
        dots_pos = [
            Dot(
                point=plane.coords_to_point(x[0], x[1]),
                color=RED if y[i] == 0 else BLUE,
                radius=0.1,
                stroke_width=5,
                fill_opacity=0,
            )
            for i, x in enumerate(X)
        ]
        legend_labels = [Text("Class 0", color=RED), Text("Class 1", color=BLUE)]
        legend = VGroup(*legend_labels).arrange(DOWN).to_corner(UL)

        self.play(*[Create(dot) for dot in dots_pos], Create(legend))
        self.wait(0.5)

        # Initialize SVM classifier
        svm = SVC(kernel="linear", C=1, max_iter=1, random_state=42)

        # Iterate over different numbers of iterations
        for n_iter in range(1, 11):
            svm.set_params(max_iter=n_iter, random_state=42)
            svm.fit(X, y)

            # Get support vectors, coefficients, and intercept
            support_vectors = svm.support_vectors_
            coef = svm.coef_[0]
            intercept = svm.intercept_[0]

            # Highlight support vectors
            support_vector_dots = VGroup(
                *[
                    Dot(
                        point=plane.coords_to_point(x[0], x[1]),
                        color=YELLOW,
                        radius=0.15,
                        stroke_width=5,
                        fill_opacity=0,
                    )
                    for x in support_vectors
                ]
            )

            # Calculate decision boundary coordinates
            xmin, xmax = plane.x_range[0], plane.x_range[1]
            ymin = (-coef[0] * xmin - intercept) / coef[1]
            ymax = (-coef[0] * xmax - intercept) / coef[1]
            norm = np.linalg.norm(coef)
            dist = 1 / (coef[1] * norm)

            # Create decision boundary and margins
            decision_boundary_new = Line(
                plane.coords_to_point(xmin, ymin), plane.coords_to_point(xmax, ymax), color=GREEN
            )
            margin_pos_new = Line(
                plane.coords_to_point(xmin, ymin + dist),
                plane.coords_to_point(xmax, ymax + dist),
                color=YELLOW,
                stroke_opacity=0.5,
            )
            margin_neg_new = Line(
                plane.coords_to_point(xmin, ymin - dist),
                plane.coords_to_point(xmax, ymax - dist),
                color=YELLOW,
                stroke_opacity=0.5,
            )

            iteration_text_new = Tex(f"Iteration: {n_iter}").move_to([3, 3, 0])

            # Predict class labels for all points in the feature space
            y_pred = svm.predict(X)

            dots_pred_new = VGroup(
                *[
                    Dot(point=plane.coords_to_point(x[0], x[1]), color=RED if yp == 0 else BLUE, radius=0.05)
                    for x, yp in zip(X, y_pred)
                ]
            )

            # Animate support vectors
            self.play(Create(support_vector_dots))

            # Animate decision boundary, margins, and classification
            if n_iter == 1:
                self.play(
                    Write(iteration_text_new),
                    Create(decision_boundary_new),
                    Create(margin_pos_new),
                    Create(margin_neg_new),
                )
                self.play(Create(dots_pred_new))

                decision_boundary_last = decision_boundary_new
                margin_pos_last = margin_pos_new
                margin_neg_last = margin_neg_new
                iteration_text_last = iteration_text_new
                dots_pred_last = dots_pred_new

            else:
                self.play(
                    Transform(iteration_text_last, iteration_text_new),
                    Transform(decision_boundary_last, decision_boundary_new),
                    Transform(margin_pos_last, margin_pos_new),
                    Transform(margin_neg_last, margin_neg_new),
                )

                self.play(Transform(dots_pred_last, dots_pred_new))

            self.wait(0.2)
            self.remove(support_vector_dots)

        self.wait(2)
