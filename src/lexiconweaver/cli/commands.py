"""CLI commands for LexiconWeaver."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lexiconweaver.config import Config
from lexiconweaver.database import (
    GlossaryTerm,
    IgnoredTerm,
    Project,
    close_database,
    initialize_database,
)
from lexiconweaver.engines.scout import Scout
from lexiconweaver.engines.weaver import Weaver
from lexiconweaver.exceptions import LexiconWeaverError
from lexiconweaver.logging_config import configure_logging, get_logger
from lexiconweaver.tui.app import LexiconWeaverApp
from lexiconweaver.utils.validators import validate_text_file

app = typer.Typer(name="lexiconweaver", help="LexiconWeaver: Web Novel Translation Framework")
console = Console()
logger = get_logger(__name__)


def get_config() -> Config:
    """Load configuration."""
    return Config.load()


def get_project(project_name: Optional[str] = None) -> Project:
    """Get or create a project."""
    if project_name:
        project, created = Project.get_or_create(title=project_name)
        if created:
            console.print(f"[green]Created project: {project_name}[/green]")
        return project
    else:
        # Get the most recent project or create a default one
        project = Project.select().order_by(Project.created_at.desc()).first()
        if project is None:
            project = Project.create(title="default")
            console.print(f"[yellow]Created default project[/yellow]")
        return project


@app.command()
def project(
    action: str = typer.Argument(..., help="Action: create, list, select, delete"),
    name: Optional[str] = typer.Argument(None, help="Project name"),
):
    """Manage translation projects."""
    try:
        config = get_config()
        initialize_database(config)

        if action == "create":
            if not name:
                console.print("[red]Error: Project name required for create action[/red]")
                raise typer.Exit(1)
            project, created = Project.get_or_create(title=name)
            if created:
                console.print(f"[green]Created project: {name}[/green]")
            else:
                console.print(f"[yellow]Project already exists: {name}[/yellow]")

        elif action == "list":
            projects = Project.select()
            if not projects:
                console.print("[yellow]No projects found[/yellow]")
                return

            table = Table(title="Projects")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Source Lang", style="yellow")
            table.add_column("Target Lang", style="yellow")
            table.add_column("Created", style="blue")

            for proj in projects:
                table.add_row(
                    str(proj.id),
                    proj.title,
                    proj.source_lang,
                    proj.target_lang,
                    str(proj.created_at),
                )
            console.print(table)

        elif action == "select":
            if not name:
                console.print("[red]Error: Project name required for select action[/red]")
                raise typer.Exit(1)
            try:
                project = Project.get(Project.title == name)
                console.print(f"[green]Selected project: {name}[/green]")
            except Project.DoesNotExist:
                console.print(f"[red]Project not found: {name}[/red]")
                raise typer.Exit(1)

        elif action == "delete":
            if not name:
                console.print("[red]Error: Project name required for delete action[/red]")
                raise typer.Exit(1)
            try:
                project = Project.get(Project.title == name)
                project.delete_instance(recursive=True)
                console.print(f"[green]Deleted project: {name}[/green]")
            except Project.DoesNotExist:
                console.print(f"[red]Project not found: {name}[/red]")
                raise typer.Exit(1)

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            raise typer.Exit(1)

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        close_database()


@app.command()
def scout(
    file: Path = typer.Argument(..., help="Text file to analyze"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    min_confidence: float = typer.Option(0.3, "--min-confidence", help="Minimum confidence score"),
):
    """Run Scout engine to discover potential terms."""
    try:
        config = get_config()
        config.scout.min_confidence = min_confidence
        configure_logging(config)
        initialize_database(config)

        project = get_project(project_name)

        # Validate file
        file_path, encoding = validate_text_file(file)

        # Read file
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()

        # Run Scout
        with console.status("[bold green]Running Scout..."):
            scout = Scout(config, project)
            candidates = scout.process(text)

        # Display results
        if not candidates:
            console.print("[yellow]No candidate terms found[/yellow]")
            return

        table = Table(title=f"Candidate Terms (Found: {len(candidates)})")
        table.add_column("Term", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Frequency", style="yellow")
        table.add_column("Pattern", style="blue")

        for candidate in candidates:
            table.add_row(
                candidate.term,
                f"{candidate.confidence:.2f}",
                str(candidate.frequency),
                candidate.context_pattern or "-",
            )

        console.print(table)

        # Save to file if specified
        if output:
            with open(output, "w", encoding="utf-8") as f:
                for candidate in candidates:
                    f.write(
                        f"{candidate.term}\t{candidate.confidence:.2f}\t{candidate.frequency}\t{candidate.context_pattern or '-'}\n"
                    )
            console.print(f"[green]Results saved to {output}[/green]")

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Scout command failed")
        raise typer.Exit(1)
    finally:
        close_database()


@app.command()
def translate(
    file: Path = typer.Argument(..., help="Text file to translate"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for translation"),
):
    """Translate text using Weaver engine."""
    try:
        config = get_config()
        configure_logging(config)
        initialize_database(config)

        project = get_project(project_name)

        # Validate file
        file_path, encoding = validate_text_file(file)

        # Read file
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()

        # Translate
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Translating...", total=None)
            weaver = Weaver(config, project)
            translation = asyncio.run(weaver.translate_text(text))
            progress.update(task, completed=True)

        # Output
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(translation)
            console.print(f"[green]Translation saved to {output}[/green]")
        else:
            console.print("\n[bold]Translation:[/bold]")
            console.print(translation)

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Translate command failed")
        raise typer.Exit(1)
    finally:
        close_database()


@app.command()
def tui(
    file: Optional[Path] = typer.Argument(None, help="Text file to open"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
):
    """Launch the Textual TUI interface."""
    try:
        config = get_config()
        configure_logging(config, quiet=True)  # Suppress console output in TUI mode
        initialize_database(config)

        project = get_project(project_name)

        text = ""
        if file:
            file_path, encoding = validate_text_file(file)
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            text_file = file_path
        else:
            text_file = None

        app = LexiconWeaverApp(config, project, text=text, text_file=text_file)
        app.run()

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("TUI launch failed")
        raise typer.Exit(1)
    finally:
        close_database()


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
