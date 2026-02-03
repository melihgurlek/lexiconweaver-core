"""CLI commands for LexiconWeaver."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from lexiconweaver.config import Config
from lexiconweaver.database import (
    GlossaryTerm,
    IgnoredTerm,
    Project,
    close_database,
    initialize_database,
)
from lexiconweaver.database.models import ProposedTerm
from lexiconweaver.engines.scout import Scout
from lexiconweaver.engines.scout_refiner import ScoutRefiner
from lexiconweaver.engines.weaver import Weaver
from lexiconweaver.exceptions import LexiconWeaverError
from lexiconweaver.logging_config import configure_logging, get_logger
from lexiconweaver.providers import LLMProviderManager
from lexiconweaver.tui.app import LexiconWeaverApp
from lexiconweaver.utils.document_loader import load_document
from lexiconweaver.utils.document_writer import write_document

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
        project = Project.select().order_by(Project.created_at.desc()).first()
        if project is None:
            project = Project.create(title="default")
            console.print(f"[yellow]Created default project[/yellow]")
        return project


@app.command("test-connection")
def test_connection(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Run a minimal generate test"),
):
    """Test LLM provider connection (Ollama and/or DeepSeek)."""
    try:
        config = get_config()
        configure_logging(config)
        # Don't initialize database for connection test

        console.print("[bold]LexiconWeaver Connection Test[/bold]\n")

        # Show config (without exposing API key)
        console.print("[cyan]Provider config:[/cyan]")
        console.print(f"  Primary: [bold]{config.provider.primary}[/bold]")
        console.print(f"  Fallback: {config.provider.fallback}")
        console.print(f"  Fallback on error: {config.provider.fallback_on_error}")
        console.print()

        if config.provider.primary == "ollama":
            console.print(f"  Ollama URL: {config.ollama.url}")
            console.print(f"  Ollama model: {config.ollama.model}")
        else:
            console.print(f"  DeepSeek base URL: {config.deepseek.base_url}")
            console.print(f"  DeepSeek model: {config.deepseek.model}")
            key_set = bool((config.deepseek.api_key or "").strip())
            console.print(f"  DeepSeek API key: {'[green]set[/green]' if key_set else '[red]not set[/red]'}")
        console.print()

        # Test availability
        async def _run_test():
            manager = LLMProviderManager(config)
            console.print("[cyan]Checking provider availability...[/cyan]")
            status = await manager.check_availability()
            for provider_name, available in status.items():
                if available:
                    console.print(f"  [green]✓[/green] {provider_name}: [green]available[/green]")
                else:
                    console.print(f"  [red]✗[/red] {provider_name}: [red]not available[/red]")
            console.print()

            # Optionally run a minimal generate
            if verbose:
                console.print("[cyan]Running minimal generate test...[/cyan]")
                try:
                    provider = await manager.get_available_provider()
                    console.print(f"  Using provider: [bold]{provider.name}[/bold]")
                    messages = [
                        {"role": "user", "content": "Reply with exactly: OK"},
                    ]
                    response = await manager.generate(messages)
                    response_preview = (response or "").strip()[:80]
                    console.print(f"  [green]✓[/green] Response: {response_preview!r}")
                except Exception as e:
                    console.print(f"  [red]✗[/red] Generate failed: {e}")

        asyncio.run(_run_test())
        console.print("\n[bold green]Done.[/bold green]")

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Test connection failed")
        raise typer.Exit(1)


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
    file: Path = typer.Argument(..., help="Text, EPUB, or PDF file to analyze"),
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

        text = load_document(file)

        with console.status("[bold green]Running Scout..."):
            scout_engine = Scout(config, project)
            candidates = scout_engine.process(text)

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


@app.command("smart-scout")
def smart_scout(
    file: Path = typer.Argument(..., help="Text, EPUB, or PDF file to analyze"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results (JSON)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode for term approval"),
):
    """Run Smart Scout with LLM-based term refinement and translation proposals."""
    try:
        config = get_config()
        configure_logging(config)
        initialize_database(config)

        project = get_project(project_name)

        text = load_document(file)

        console.print("[bold cyan]Smart Scout: Two-pass term discovery with LLM[/bold cyan]")
        console.print(f"Provider: {config.provider.primary}")
        if config.provider.fallback != "none":
            console.print(f"Fallback: {config.provider.fallback}")
        console.print()

        refined_terms = asyncio.run(_run_smart_scout(config, project, text))

        if not refined_terms:
            console.print("[yellow]No terms found after LLM refinement[/yellow]")
            return

        console.print(f"\n[bold green]Found {len(refined_terms)} terms[/bold green]\n")

        if interactive:
            approved_count = 0
            modified_count = 0
            rejected_count = 0

            for i, term in enumerate(refined_terms, 1):
                console.print(f"\n[bold]Term {i}/{len(refined_terms)}[/bold]")
                console.print(f"  [cyan]Source:[/cyan] {term.source_term}")
                console.print(f"  [green]Proposed Translation:[/green] {term.proposed_translation}")
                console.print(f"  [yellow]Category:[/yellow] {term.proposed_category or 'N/A'}")
                if term.reasoning:
                    console.print(f"  [dim]Reasoning:[/dim] {term.reasoning}")
                if term.context_snippet:
                    console.print(f"  [dim]Context:[/dim] {term.context_snippet[:100]}...")

                action = Prompt.ask(
                    "\n  Action",
                    choices=["a", "m", "r", "s", "q"],
                    default="a",
                )

                if action == "a":  # Approve
                    _save_term_to_glossary(
                        project,
                        term.source_term,
                        term.proposed_translation,
                        term.proposed_category,
                    )
                    _update_proposal_status(project, term.source_term, "approved")
                    approved_count += 1
                    console.print("  [green]Approved[/green]")

                elif action == "m":  # Modify
                    new_translation = Prompt.ask("  Enter translation", default=term.proposed_translation)
                    new_category = Prompt.ask(
                        "  Category",
                        choices=["Person", "Location", "Skill", "Clan", "Item", "Other"],
                        default=term.proposed_category or "Other",
                    )
                    _save_term_to_glossary(project, term.source_term, new_translation, new_category)
                    _update_proposal_status(
                        project, term.source_term, "modified",
                        user_translation=new_translation, user_category=new_category
                    )
                    modified_count += 1
                    console.print("  [blue]Modified and saved[/blue]")

                elif action == "r":  # Reject
                    _update_proposal_status(project, term.source_term, "rejected")
                    rejected_count += 1
                    console.print("  [red]Rejected[/red]")

                elif action == "s":  # Skip
                    console.print("  [dim]Skipped[/dim]")

                elif action == "q":  # Quit
                    console.print("\n[yellow]Quitting interactive mode[/yellow]")
                    break

            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Approved: {approved_count}")
            console.print(f"  Modified: {modified_count}")
            console.print(f"  Rejected: {rejected_count}")

        else:
            table = Table(title="Smart Scout Results")
            table.add_column("Term", style="cyan")
            table.add_column("Proposed Translation", style="green")
            table.add_column("Category", style="yellow")
            table.add_column("Reasoning", style="dim", max_width=40)

            for term in refined_terms:
                table.add_row(
                    term.source_term,
                    term.proposed_translation,
                    term.proposed_category or "-",
                    (term.reasoning[:37] + "...") if term.reasoning and len(term.reasoning) > 40 else (term.reasoning or "-"),
                )

            console.print(table)
            console.print(f"\n[dim]Use --interactive (-i) to approve terms one by one[/dim]")
            console.print(f"[dim]Or use 'approve-terms' command to process pending proposals[/dim]")

        if output:
            output_data = [
                {
                    "term": t.source_term,
                    "translation": t.proposed_translation,
                    "category": t.proposed_category,
                    "reasoning": t.reasoning,
                    "context": t.context_snippet,
                }
                for t in refined_terms
            ]
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Results saved to {output}[/green]")

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Smart Scout command failed")
        raise typer.Exit(1)
    finally:
        close_database()


async def _run_smart_scout(config: Config, project: Project, text: str):
    """Run the smart scout asynchronously."""
    refiner = ScoutRefiner(config, project)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Pass 1: Running heuristic scout...", total=None)
        
        progress.update(task, description="Pass 2: LLM refinement and proposals...")
        refined_terms = await refiner.refine_text(text)
        
        progress.update(task, description="Saving proposals to database...")
        await refiner.save_proposals(refined_terms)
        
        progress.update(task, completed=True, description="Complete!")
    
    return refined_terms


def _save_term_to_glossary(project: Project, source: str, target: str, category: str | None) -> None:
    """Save a term to the glossary."""
    term, created = GlossaryTerm.get_or_create(
        project=project,
        source_term=source,
        defaults={
            "target_term": target,
            "category": category,
        },
    )
    if not created:
        term.target_term = target
        term.category = category
        term.save()


def _update_proposal_status(
    project: Project,
    source_term: str,
    status: str,
    user_translation: str | None = None,
    user_category: str | None = None,
) -> None:
    """Update the status of a proposed term."""
    try:
        proposal = ProposedTerm.get_or_none(
            ProposedTerm.project == project,
            ProposedTerm.source_term == source_term,
        )
        if proposal:
            proposal.status = status
            if user_translation:
                proposal.user_translation = user_translation
            if user_category:
                proposal.user_category = user_category
            proposal.save()
    except Exception as e:
        logger.debug("Error updating proposal status", error=str(e))


@app.command("approve-terms")
def approve_terms(
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    interactive: bool = typer.Option(True, "--interactive/--batch", "-i/-b", help="Interactive or batch mode"),
    approve_all: bool = typer.Option(False, "--approve-all", "-a", help="Approve all pending terms (batch mode)"),
):
    """Review and approve pending term proposals."""
    try:
        config = get_config()
        configure_logging(config)
        initialize_database(config)

        project = get_project(project_name)

        pending = list(ProposedTerm.select().where(
            ProposedTerm.project == project,
            ProposedTerm.status == "pending",
        ))

        if not pending:
            console.print("[yellow]No pending proposals found[/yellow]")
            return

        console.print(f"[bold]Found {len(pending)} pending proposals[/bold]\n")

        if approve_all:
            if not Confirm.ask(f"Approve all {len(pending)} proposals?"):
                console.print("[yellow]Cancelled[/yellow]")
                return

            for proposal in pending:
                _save_term_to_glossary(
                    project,
                    proposal.source_term,
                    proposal.proposed_translation,
                    proposal.proposed_category,
                )
                proposal.status = "approved"
                proposal.save()

            console.print(f"[green]Approved {len(pending)} terms[/green]")

        elif interactive:
            approved = 0
            modified = 0
            rejected = 0

            for i, proposal in enumerate(pending, 1):
                console.print(f"\n[bold]Proposal {i}/{len(pending)}[/bold]")
                console.print(f"  [cyan]Term:[/cyan] {proposal.source_term}")
                console.print(f"  [green]Translation:[/green] {proposal.proposed_translation}")
                console.print(f"  [yellow]Category:[/yellow] {proposal.proposed_category or 'N/A'}")
                if proposal.llm_reasoning:
                    console.print(f"  [dim]Reasoning:[/dim] {proposal.llm_reasoning}")
                if proposal.context_snippet:
                    console.print(f"  [dim]Context:[/dim] {proposal.context_snippet[:100]}...")

                action = Prompt.ask(
                    "\n  [A]pprove / [M]odify / [R]eject / [S]kip / [Q]uit",
                    choices=["a", "m", "r", "s", "q"],
                    default="a",
                )

                if action == "a":
                    _save_term_to_glossary(
                        project,
                        proposal.source_term,
                        proposal.proposed_translation,
                        proposal.proposed_category,
                    )
                    proposal.status = "approved"
                    proposal.save()
                    approved += 1
                    console.print("  [green]Approved[/green]")

                elif action == "m":
                    new_translation = Prompt.ask("  Translation", default=proposal.proposed_translation)
                    new_category = Prompt.ask(
                        "  Category",
                        choices=["Person", "Location", "Skill", "Clan", "Item", "Other"],
                        default=proposal.proposed_category or "Other",
                    )
                    _save_term_to_glossary(project, proposal.source_term, new_translation, new_category)
                    proposal.status = "modified"
                    proposal.user_translation = new_translation
                    proposal.user_category = new_category
                    proposal.save()
                    modified += 1
                    console.print("  [blue]Modified[/blue]")

                elif action == "r":
                    proposal.status = "rejected"
                    proposal.save()
                    rejected += 1
                    console.print("  [red]Rejected[/red]")

                elif action == "s":
                    console.print("  [dim]Skipped[/dim]")

                elif action == "q":
                    console.print("\n[yellow]Quitting[/yellow]")
                    break

            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Approved: {approved}")
            console.print(f"  Modified: {modified}")
            console.print(f"  Rejected: {rejected}")

        else:
            # Non-interactive: just show table
            table = Table(title="Pending Proposals")
            table.add_column("Term", style="cyan")
            table.add_column("Translation", style="green")
            table.add_column("Category", style="yellow")

            for proposal in pending:
                table.add_row(
                    proposal.source_term,
                    proposal.proposed_translation,
                    proposal.proposed_category or "-",
                )

            console.print(table)
            console.print("\n[dim]Use --interactive to approve terms one by one[/dim]")
            console.print("[dim]Or use --approve-all to approve all at once[/dim]")

    except LexiconWeaverError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.exception("Approve terms command failed")
        raise typer.Exit(1)
    finally:
        close_database()


@app.command()
def translate(
    file: Path = typer.Argument(..., help="Text, EPUB, or PDF file to translate"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file for translation (format from extension: .txt, .pdf, or .epub)",
    ),
):
    """Translate text using Weaver engine."""
    try:
        config = get_config()
        configure_logging(config)
        initialize_database(config)

        project = get_project(project_name)

        text = load_document(file)

        console.print(f"[bold cyan]Translating with provider: {config.provider.primary}[/bold cyan]")
        if config.provider.fallback != "none":
            console.print(f"[dim]Fallback: {config.provider.fallback}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Translating...", total=None)
            weaver = Weaver(config, project)
            translation = asyncio.run(weaver.translate_text(text))
            progress.update(task, completed=True)

        if output:
            write_document(output, translation, title=file.stem if file else "Translation")
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
    file: Optional[Path] = typer.Argument(None, help="Text, EPUB, or PDF file to open"),
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
):
    """Launch the Textual TUI interface."""
    try:
        config = get_config()
        configure_logging(config, quiet=True)
        initialize_database(config)

        project = get_project(project_name)

        text = ""
        text_file: Optional[Path] = None
        if file:
            text = load_document(file)
            text_file = file

        tui_app = LexiconWeaverApp(config, project, text=text, text_file=text_file)
        tui_app.run()

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
