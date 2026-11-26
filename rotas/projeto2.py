from flask import Blueprint, render_template, request

proj2_bp = Blueprint("proj2", __name__)


# ROTAS
@proj2_bp.route("/proj2")
def projeto2():
    return render_template("projeto2.html")

@proj2_bp.route("/kmeans")
def kmeans():
    return render_template("kmeans.html")